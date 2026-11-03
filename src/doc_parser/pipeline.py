"""Main document parsing pipeline wrapping the GLM-OCR SDK."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from doc_parser.config import get_settings
from doc_parser.post_processor import assemble_markdown, save_to_json
from doc_parser.utils.pdf_utils import count_pdf_pages

logger = logging.getLogger(__name__)

try:
    from glmocr import GlmOcr  # type: ignore[import]
    import glmocr.ocr_client as _ocr_client

    # ── Monkey-patch: fix empty prompt bug in glmocr SDK ─────────────────────
    # The SDK sends images to Ollama but leaves prompt="" — the model returns
    # nothing without an instruction. Inject the OCR prompt when it's missing.
    _original_convert = _ocr_client.OCRClient._convert_to_ollama_generate

    def _patched_convert(self, request_data):  # type: ignore[misc]
        result = _original_convert(self, request_data)
        if not result.get("prompt"):
            result["prompt"] = (
                "Please carefully read and transcribe ALL text visible in this image. "
                "Output only the transcribed text, preserving the original layout."
            )
        return result

    _ocr_client.OCRClient._convert_to_ollama_generate = _patched_convert

    # ── Monkey-patch: force Ollama context size + keep_alive per request ────
    # The SDK never sends num_ctx, so Ollama falls back to its default (4096).
    # Image tokens (~2-3k for a large region) + num_predict (4096) overflow
    # that, causing truncated prompts / runaway generations / read timeouts.
    # Setting it per request makes this independent of OLLAMA_CONTEXT_LENGTH.
    import os as _os
    import time as _time
    import threading as _threading

    # 10240: glm-ocr GGUF emits empty ``` fences at small contexts (4096); 10k is
    # the commonly reported working value.
    _OLLAMA_NUM_CTX = int(_os.getenv("OLLAMA_NUM_CTX", "10240"))
    _OLLAMA_KEEP_ALIVE = _os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    _convert_with_prompt = _ocr_client.OCRClient._convert_to_ollama_generate

    # glm-ocr GGUF (Ollama) often falls into repetition loops and burns the
    # whole num_predict budget (4096 tok ≈ 45s/region on an RTX 3050). A region
    # can't contain more text than its pixels allow, so cap num_predict by the
    # crop's vision-token count (~784 px per token).
    _OUT_PER_VISION_TOKEN = float(_os.getenv("OCR_OUT_PER_VISION_TOKEN", "3"))
    _MIN_NUM_PREDICT = int(_os.getenv("OCR_MIN_NUM_PREDICT", "384"))

    def _adaptive_num_predict(images, current):
        if not images:
            return current
        try:
            import base64
            from io import BytesIO
            from PIL import Image

            with Image.open(BytesIO(base64.b64decode(images[0]))) as im:
                w, h = im.size
            vision_tokens = (w * h) / 784.0
            cap = max(_MIN_NUM_PREDICT, int(vision_tokens * _OUT_PER_VISION_TOKEN) + 128)
            return min(int(current), cap) if current else cap
        except Exception:
            return current

    def _patched_convert_ctx(self, request_data):  # type: ignore[misc]
        result = _convert_with_prompt(self, request_data)
        opts = result.setdefault("options", {})
        opts["num_ctx"] = _OLLAMA_NUM_CTX
        opts["num_predict"] = _adaptive_num_predict(
            result.get("images"), opts.get("num_predict", 4096)
        )
        result["keep_alive"] = _OLLAMA_KEEP_ALIVE
        return result

    _ocr_client.OCRClient._convert_to_ollama_generate = _patched_convert_ctx

    # ── Monkey-patch: per-region progress logging ────────────────────────────
    # The SDK is silent at INFO while OCR runs; log each region so a long
    # ingest doesn't look hung.
    _original_process = _ocr_client.OCRClient.process
    _region_counter = {"n": 0}
    _region_lock = _threading.Lock()
    _last_stats = _threading.local()  # per worker thread: raw Ollama stats

    def _capture_ollama_stats(resp, *args, **kwargs):  # requests response hook
        try:
            data = resp.json()
            _last_stats.value = {
                k: data.get(k)
                for k in ("prompt_eval_count", "eval_count", "done_reason",
                          "prompt_eval_duration", "eval_duration", "load_duration")
            }
        except Exception:
            _last_stats.value = None
        return resp

    _original_make_session = _ocr_client.OCRClient._make_session

    def _patched_make_session(self):  # type: ignore[misc]
        session = _original_make_session(self)
        session.hooks["response"].append(_capture_ollama_stats)
        return session

    _ocr_client.OCRClient._make_session = _patched_make_session

    def _trim_repetition(text: str) -> str:
        """Drop the partial last line and collapse repeated trailing line cycles
        (e.g. the same line or a ``` fence repeated hundreds of times)."""
        lines = text.split("\n")
        if len(lines) > 1:
            lines = lines[:-1]  # last line was cut mid-token by num_predict
        for period in range(1, 11):
            while len(lines) >= 2 * period and lines[-period:] == lines[-2 * period:-period]:
                lines = lines[:-period]
        # strip dangling empty/fence-only lines left at the tail
        while lines and lines[-1].strip() in ("", "```"):
            lines = lines[:-1]
        return "\n".join(lines)

    def _patched_process(self, request_data):  # type: ignore[misc]
        with _region_lock:
            _region_counter["n"] += 1
            n = _region_counter["n"]
        t0 = _time.perf_counter()
        _last_stats.value = None
        response, status = _original_process(self, request_data)
        dt = _time.perf_counter() - t0
        if status == 200:
            st = getattr(_last_stats, "value", None) or {}
            ns = 1e9
            text = ""
            try:
                text = response["choices"][0]["message"]["content"] or ""
            except Exception:
                pass
            logger.info(
                "OCR region #%d done in %.1fs | in=%s tok (%.1fs) out=%s tok (%.1fs) "
                "load=%.1fs stop=%s chars=%d",
                n, dt,
                st.get("prompt_eval_count"), (st.get("prompt_eval_duration") or 0) / ns,
                st.get("eval_count"), (st.get("eval_duration") or 0) / ns,
                (st.get("load_duration") or 0) / ns,
                st.get("done_reason"), len(text),
            )
            if st.get("done_reason") == "length":
                cleaned = _trim_repetition(text)
                try:
                    response["choices"][0]["message"]["content"] = cleaned
                except Exception:
                    pass
                logger.warning(
                    "OCR region #%d hit num_predict (repetition loop) — trimmed %d -> %d chars",
                    n, len(text), len(cleaned),
                )
        else:
            logger.warning(
                "OCR region #%d failed (HTTP %s) after %.1fs: %s",
                n, status, dt, str(response.get("error", ""))[:200],
            )
        return response, status

    _ocr_client.OCRClient.process = _patched_process
    # ─────────────────────────────────────────────────────────────────────────

    _GLMOCR_AVAILABLE = True
except ImportError:
    _GLMOCR_AVAILABLE = False
    logger.warning(
        "glmocr package not installed. Install with: uv pip install glmocr"
    )


@dataclass
class ParsedElement:
    """A single detected and recognized document element.

    Attributes:
        label: Element category (e.g. 'document_title', 'paragraph', 'table').
        text: Recognized text content.
        bbox: Bounding box [x1, y1, x2, y2] in normalized coordinates.
        score: Detection confidence score (0.0–1.0).
        reading_order: Sequence index for correct reading order assembly.
    """

    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int


@dataclass
class PageResult:
    """Parsing result for a single document page.

    Attributes:
        page_num: One-based page number.
        elements: All detected elements on this page.
        markdown: Assembled Markdown string for this page.
    """

    page_num: int
    elements: list[ParsedElement] = field(default_factory=list)
    markdown: str = ""


@dataclass
class ParseResult:
    """Complete parsing result for a document.

    Attributes:
        source_file: Path to the original document.
        pages: Per-page parsing results.
        total_elements: Sum of all elements across all pages.
    """

    source_file: str
    pages: list[PageResult] = field(default_factory=list)
    total_elements: int = 0
    full_markdown: str = ""  # Full document markdown from SDK (preferred over per-page assembly)

    @classmethod
    def from_sdk_result(cls, raw: Any, source_file: str) -> ParseResult:
        """Build a ParseResult from a raw GLM-OCR SDK PipelineResult.

        The SDK returns a PipelineResult with:
        - json_result: list[list[dict]] — one inner list per page, each dict has
          'index', 'label', 'content', 'bbox_2d'
        - markdown_result: str — full document markdown for the entire document

        Args:
            raw: PipelineResult from GlmOcr.parse().
            source_file: Path to the source document.

        Returns:
            Populated ParseResult instance.
        """
        pages: list[PageResult] = []

        # json_result is a list-of-lists: [page][element]
        raw_pages: list[list[dict[str, Any]]] = getattr(raw, "json_result", [])

        # markdown_result is a single string for the whole document (not per-page)
        full_markdown: str = getattr(raw, "markdown_result", "") or ""

        for page_idx, raw_elements in enumerate(raw_pages):
            page_num = page_idx + 1
            elements: list[ParsedElement] = []

            for raw_el in raw_elements:
                bbox_2d = raw_el.get("bbox_2d", [0, 0, 1, 1])
                el = ParsedElement(
                    label=raw_el.get("label", "paragraph"),
                    text=raw_el.get("content", ""),
                    bbox=[float(v) for v in bbox_2d],
                    score=1.0,  # SDK does not provide a confidence score
                    reading_order=raw_el.get("index", len(elements)),
                )
                elements.append(el)

            # Per-page markdown assembled from elements (used for chunking metadata)
            markdown = assemble_markdown(elements)  # type: ignore[arg-type]
            pages.append(PageResult(page_num=page_num, elements=elements, markdown=markdown))

        total_elements = sum(len(p.elements) for p in pages)
        return cls(
            source_file=source_file,
            pages=pages,
            total_elements=total_elements,
            full_markdown=full_markdown,
        )

    def save(self, output_dir: Path) -> None:
        """Save this result as Markdown and JSON files.

        Args:
            output_dir: Directory to write output files.
        """
        save_to_json(self, output_dir)


class DocumentParser:
    """Main document parsing pipeline using GLM-OCR MaaS API.

    Wraps the glmocr SDK to provide structured document parsing
    via the Z.AI cloud API (PP-DocLayout-V3 + GLM-OCR 0.9B).

    Example:
        parser = DocumentParser()
        result = parser.parse_file(Path("document.pdf"))
        result.save(Path("./output"))
    """

    def __init__(self) -> None:
        """Initialize the DocumentParser with settings from environment."""
        if not _GLMOCR_AVAILABLE:
            raise ImportError(
                "glmocr package is required. Install with: uv pip install glmocr"
            )
        settings = get_settings()
        # Only pass api_key in cloud mode. In Ollama/self-hosted mode, passing
        # any key (even a valid one) causes GlmOcr to override the YAML config
        # and force MaaS mode regardless of maas.enabled=false.
        api_key = (
            settings.z_ai_api_key.get_secret_value()
            if settings.parser_backend == "cloud" and settings.z_ai_api_key
            else None
        )
        self._parser = GlmOcr(
            config_path=settings.config_yaml_path,
            api_key=api_key,
        )
        logger.info("DocumentParser initialized with config: %s", settings.config_yaml_path)

    def parse_file(self, file_path: str | Path) -> ParseResult:
        """Parse a single PDF or image file.

        Args:
            file_path: Path to the document to parse (PDF or image).

        Returns:
            ParseResult with structured elements and assembled Markdown.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If glmocr is not installed.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Parsing file: %s", file_path)

        settings = get_settings()
        parse_kwargs: dict[str, Any] = {}
        if file_path.suffix.lower() == ".pdf":
            total_pages = count_pdf_pages(file_path)
            if settings.parser_backend == "cloud":
                # Cloud/MaaS mode: must pass explicit page range or the SDK
                # defaults to page 1 only.
                parse_kwargs["start_page_id"] = 0
                parse_kwargs["end_page_id"] = total_pages - 1
                logger.info("PDF has %d pages (PyMuPDF) — parsing all via MaaS", total_pages)
            else:
                # Ollama/self-hosted mode: the glmocr SDK silently drops
                # start_page_id/end_page_id — the internal pypdfium2 page
                # loader handles page range itself.  PyMuPDF and pypdfium2 may
                # report different page counts for certain PDFs; the count below
                # is informational only.
                logger.info(
                    "PDF has %d pages (PyMuPDF) — Ollama mode uses pypdfium2 "
                    "internally; page count may differ slightly",
                    total_pages,
                )

        if settings.parser_backend == "ollama":
            parse_kwargs["save_layout_visualization"] = False

        raw = self._parser.parse(str(file_path), **parse_kwargs)
        result = ParseResult.from_sdk_result(raw, source_file=str(file_path))
        if file_path.suffix.lower() == ".pdf" and len(result.pages) != total_pages:
            logger.warning(
                "Parsed %s: glmocr returned %d pages but PyMuPDF counted %d. "
                "The PDF may have a blank/unreadable trailing page or pypdfium2 "
                "and PyMuPDF disagree on its structure.",
                file_path.name,
                len(result.pages),
                total_pages,
            )
        else:
            logger.info(
                "Parsed %s: %d pages, %d elements",
                file_path.name,
                len(result.pages),
                result.total_elements,
            )
        return result

    def parse_batch(
        self,
        file_paths: list[Path],
        output_dir: Path,
    ) -> list[ParseResult]:
        """Parse multiple files with progress tracking.

        Args:
            file_paths: List of paths to documents to parse.
            output_dir: Directory to save output files.

        Returns:
            List of ParseResult objects, one per input file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ParseResult] = []

        for fp in tqdm(file_paths, desc="Parsing documents", unit="file"):
            try:
                result = self.parse_file(fp)
                result.save(output_dir)
                results.append(result)
            except Exception as e:
                logger.error("Failed to parse %s: %s", fp, e, exc_info=True)
                raise

        return results
