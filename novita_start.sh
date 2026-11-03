#!/usr/bin/env bash
# =============================================================================
# novita_start.sh — run multi-modal-rag (qwen branch) in ONE container on ONE L40S
#
# Replaces docker-compose.gpu.yml for container-based GPU clouds (Novita, Vast.ai, RunPod)
# where Docker Compose can't run. Same services, same GPU split:
#
#   qdrant     vector DB (native binary)                 CPU      :6333 (internal)
#   vllm-ocr   GLM-OCR 0.9B, original weights + MTP      0.14 GPU :18090 (internal)
#   vllm-llm   Qwen3-VL-8B-Instruct-FP8                  0.50 GPU :8001 (internal)
#   app        doc-parser FastAPI (embed/rerank/layout)  ~13 GB   :8000 (expose)
#
# Start order is enforced: qdrant + vllm-ocr -> vllm-llm -> app.
# Processes run under supervisord (auto-restart, logs in $DATA_ROOT/logs).
#
# Idempotent: first run installs everything (~10–20 min + model downloads);
# later runs (after a restart) skip installs and just start the services.
#
# Usage (inside the instance, as root):
#   export HF_TOKEN=hf_xxx            # recommended (download rate limits)
#   export GITHUB_TOKEN=ghp_xxx       # only if the repo is private
#   bash novita_start.sh              # setup + start (returns when launched)
#   bash novita_start.sh status       # service status + health of each endpoint
#   bash novita_start.sh health       # just the health checks
#   bash novita_start.sh logs         # follow all logs
#   bash novita_start.sh stop         # stop all services
# =============================================================================
set -euo pipefail

# ── Settings (override with env vars) ────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/Sivaneni/multi-modal-rag.git}"
REPO_BRANCH="${REPO_BRANCH:-qwen}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
HF_TOKEN="${HF_TOKEN:-}"
VLLM_API_KEY="${VLLM_API_KEY:-local-token}"

VLLM_SPEC="${VLLM_SPEC:-vllm==0.19.*}"            # GLM-OCR needs vLLM >= 0.19
QDRANT_VERSION="${QDRANT_VERSION:-v1.17.0}"         # matches qdrant-client 1.17
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

OCR_GPU_UTIL="${OCR_GPU_UTIL:-0.14}"
LLM_GPU_UTIL="${LLM_GPU_UTIL:-0.50}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
LLM_SERVED_NAME="${LLM_SERVED_NAME:-qwen3-vl-8b}"
LLM_MAX_SEQS="${LLM_MAX_SEQS:-32}"
OCR_MAX_SEQS="${OCR_MAX_SEQS:-128}"
HF_OFFLINE="${HF_OFFLINE:-1}"      # 1 = services never hit the Hub (models pre-downloaded below)
APP_PORT="${APP_PORT:-8000}"
OCR_PORT="${OCR_PORT:-18090}"   # not 8080: Vast/Jupyter portals use it
LLM_PORT="${LLM_PORT:-8001}"

# Persistent root: use a mounted network volume if you have one (set DATA_ROOT),
# otherwise /workspace if present, else /root/mmrag.
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -d /workspace && -w /workspace ]]; then DATA_ROOT=/workspace/mmrag; else DATA_ROOT=/root/mmrag; fi
fi

APP_DIR="$DATA_ROOT/multi-modal-rag"
VENV_VLLM="$DATA_ROOT/venv-vllm"
VENV_APP="$DATA_ROOT/venv-app"
BIN_DIR="$DATA_ROOT/bin"
RUN_DIR="$DATA_ROOT/run"
CONF_DIR="$DATA_ROOT/config"
LOG_DIR="$DATA_ROOT/logs"
QDRANT_DIR="$DATA_ROOT/qdrant"
export HF_HOME="$DATA_ROOT/hf-cache"
SUPERVISOR_CONF="$CONF_DIR/supervisord.conf"

log() { printf '\033[1;36m[novita]\033[0m %s\n' "$*"; }

# ── Sub-commands ─────────────────────────────────────────────────────────────
health_check() {
  for svc in "qdrant 6333/healthz" "vllm-ocr $OCR_PORT/health" "vllm-llm $LLM_PORT/health" "app $APP_PORT/health"; do
    set -- $svc
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$2" || true)
    printf "  %-9s %s\n" "$1" "$([[ $code == 200 ]] && echo "OK (200)" || echo "not ready ($code)")"
  done
}

case "${1:-start}" in
  status) supervisorctl -c "$SUPERVISOR_CONF" status; echo; health_check; exit 0 ;;
  health) health_check; exit 0 ;;
  logs)   exec tail -n 50 -F "$LOG_DIR"/*.log ;;
  stop)   supervisorctl -c "$SUPERVISOR_CONF" shutdown || true; exit 0 ;;
  start)  ;;
  *) echo "usage: $0 [start|status|health|logs|stop]"; exit 1 ;;
esac

mkdir -p "$DATA_ROOT" "$BIN_DIR" "$RUN_DIR" "$CONF_DIR" "$LOG_DIR" "$QDRANT_DIR" "$HF_HOME"
log "DATA_ROOT=$DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "No GPU visible — pick an L40S instance."; exit 1; }

# ── 1. System packages ───────────────────────────────────────────────────────
if ! command -v supervisord >/dev/null || ! command -v git >/dev/null; then
  log "Installing system packages..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    git curl ca-certificates supervisor libgl1 libglib2.0-0 >/dev/null
fi

# ── 2. uv (fast Python/package manager) ──────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

# ── 3. vLLM venv (separate from the app: different torch/transformers pins) ──
if [[ ! -f "$VENV_VLLM/.done" ]]; then
  log "Creating vLLM venv ($VLLM_SPEC)..."
  uv venv --python 3.12 "$VENV_VLLM"
  VIRTUAL_ENV="$VENV_VLLM" uv pip install "$VLLM_SPEC" "transformers>=5.3.0"
  touch "$VENV_VLLM/.done"
fi

# ── 4. App code + venv ───────────────────────────────────────────────────────
CLONE_URL="$REPO_URL"
if [[ -n "$GITHUB_TOKEN" ]]; then CLONE_URL="${REPO_URL/https:\/\//https://${GITHUB_TOKEN}@}"; fi
if [[ ! -d "$APP_DIR/.git" ]]; then
  log "Cloning $REPO_URL ($REPO_BRANCH)..."
  git clone --branch "$REPO_BRANCH" --depth 1 "$CLONE_URL" "$APP_DIR"
else
  log "Updating repo..."
  git -C "$APP_DIR" pull --ff-only || log "git pull skipped (local changes?)"
fi

if [[ ! -f "$VENV_APP/.done" ]]; then
  log "Creating app venv (torch cu128 + doc-parser[qwen,layout])..."
  uv venv --python 3.12 "$VENV_APP"
  VIRTUAL_ENV="$VENV_APP" uv pip install torch torchvision --index-url "$TORCH_INDEX"
  VIRTUAL_ENV="$VENV_APP" uv pip install -e "$APP_DIR[qwen,layout]" \
    --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match
  touch "$VENV_APP/.done"
fi

# ── 5. Qdrant binary ─────────────────────────────────────────────────────────
if [[ ! -x "$BIN_DIR/qdrant" ]]; then
  log "Downloading Qdrant $QDRANT_VERSION..."
  curl -fsSL "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz" \
    | tar -xz -C "$BIN_DIR" qdrant
  chmod +x "$BIN_DIR/qdrant"
fi

# ── 6. GLM-OCR SDK config (vLLM, OpenAI API) ─────────────────────────────────
cat > "$CONF_DIR/glmocr.vllm.yaml.tmpl" <<'YAML'
pipeline:
  maas:
    enabled: false
  ocr_api:
    api_host: 127.0.0.1
    api_port: __OCR_PORT__
    api_path: /v1/chat/completions
    api_mode: openai
    model: glm-ocr                 # = vllm-ocr --served-model-name
    connect_timeout: 900
    request_timeout: 300
    retry_max_attempts: 2
    retry_status_codes: [429, 500, 502, 503, 504]
    connection_pool_size: 64       # >= max_workers
  max_workers: 32                  # parallel regions per document (vLLM batches them)
  page_maxsize: 100
  region_maxsize: 0                # must be int
  page_loader:
    max_tokens: 8192
    temperature: 0.0
    top_p: 0.00001
    top_k: 1
    repetition_penalty: 1.1
    image_format: JPEG
    task_prompt_mapping:           # keys = task names below
      text: "Text Recognition:"
      table: "Table Recognition:"
      formula: "Formula Recognition:"
    pdf_dpi: 200
    pdf_max_pages: null
  result_formatter:
    output_format: both
    enable_merge_formula_numbers: true
    enable_merge_text_blocks: true
    enable_format_bullet_points: true
  enable_layout: true
  layout:
    model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors
    threshold: 0.3
    batch_size: 1
    workers: 1
    device: cuda
    use_polygon: false
    layout_nms: true
    id2label: null
    label_task_mapping:
      text: [abstract, algorithm, content, doc_title, figure_title, paragraph_title,
             reference_content, text, vertical_text, vision_footnote, seal,
             formula_number, footnote, aside_text, reference]
      table: [table]
      formula: [display_formula, inline_formula]
      skip: [chart, image]
      abandon: [header, footer, number, footer_image, header_image]
YAML
sed "s/__OCR_PORT__/$OCR_PORT/" "$CONF_DIR/glmocr.vllm.yaml.tmpl" > "$CONF_DIR/glmocr.vllm.yaml"

# ── 6b. Pre-download all models (visible progress, retry without xet) ───────
# vLLM downloads silently inside a log file and can appear to hang; doing it
# here shows progress and lets services start offline.
HF_CLI="$VENV_VLLM/bin/hf"; [[ -x "$HF_CLI" ]] || HF_CLI="$VENV_VLLM/bin/huggingface-cli"
MODELS=(
  zai-org/GLM-OCR
  "$LLM_MODEL"
  Qwen/Qwen3-VL-Embedding-2B
  Qwen/Qwen3-VL-Reranker-2B
  PaddlePaddle/PP-DocLayoutV3_safetensors
)
export HF_TOKEN
for m in "${MODELS[@]}"; do
  log "Model: $m"
  # progress bar shown live; 30-min cap so a stalled transfer can't hang forever
  if ! HF_HUB_OFFLINE=0 timeout 1800 "$HF_CLI" download "$m" >/dev/null; then
    log "  retrying $m with HF_HUB_DISABLE_XET=1 ..."
    HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 timeout 1800 "$HF_CLI" download "$m" >/dev/null || {
      echo "Download failed for $m (network/token?). Re-run the script to resume."; exit 1; }
  fi
done
log "All models cached in $HF_HOME ($(du -sh "$HF_HOME" | cut -f1))"

# ── 7. Run scripts (one per service; supervisor just execs them) ─────────────
cat > "$BIN_DIR/wait_for.sh" <<'SH'
#!/usr/bin/env bash
# wait_for.sh <url> [timeout_s] — block until URL returns 2xx
url="$1"; timeout="${2:-1800}"; start=$(date +%s)
until curl -fs -o /dev/null "$url"; do
  (( $(date +%s) - start > timeout )) && { echo "timeout waiting for $url"; exit 1; }
  sleep 5
done
SH
chmod +x "$BIN_DIR/wait_for.sh"

cat > "$RUN_DIR/qdrant.sh" <<SH
#!/usr/bin/env bash
export QDRANT__STORAGE__STORAGE_PATH="$QDRANT_DIR/storage"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$QDRANT_DIR/snapshots"
export QDRANT__SERVICE__HOST=127.0.0.1
export QDRANT__TELEMETRY_DISABLED=true
cd "$QDRANT_DIR" && exec "$BIN_DIR/qdrant"
SH

cat > "$RUN_DIR/vllm-ocr.sh" <<SH
#!/usr/bin/env bash
export HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" HF_HUB_OFFLINE="$HF_OFFLINE"
exec "$VENV_VLLM/bin/vllm" serve zai-org/GLM-OCR \\
  --served-model-name glm-ocr \\
  --host 127.0.0.1 --port $OCR_PORT \\
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \\
  --gpu-memory-utilization $OCR_GPU_UTIL \\
  --max-model-len 16384 \\
  --max-num-seqs $OCR_MAX_SEQS \\
  --max-num-batched-tokens 16384
SH

# Waits for vllm-ocr: two engines profiling free VRAM at once can over-commit.
cat > "$RUN_DIR/vllm-llm.sh" <<SH
#!/usr/bin/env bash
"$BIN_DIR/wait_for.sh" http://127.0.0.1:$OCR_PORT/health || exit 1
export HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" HF_HUB_OFFLINE="$HF_OFFLINE"
exec "$VENV_VLLM/bin/vllm" serve $LLM_MODEL \\
  --served-model-name $LLM_SERVED_NAME \\
  --host 127.0.0.1 --port $LLM_PORT \\
  --api-key $VLLM_API_KEY \\
  --gpu-memory-utilization $LLM_GPU_UTIL \\
  --max-model-len 16384 \\
  --max-num-seqs $LLM_MAX_SEQS \\
  --max-num-batched-tokens 8192 \\
  --enable-prefix-caching \\
  --limit-mm-per-prompt '{"image": 6, "video": 0}'
SH

# App starts last so the vLLM engines have already claimed their VRAM.
cat > "$RUN_DIR/app.sh" <<SH
#!/usr/bin/env bash
"$BIN_DIR/wait_for.sh" http://127.0.0.1:6333/healthz  || exit 1
"$BIN_DIR/wait_for.sh" http://127.0.0.1:$OCR_PORT/health   || exit 1
"$BIN_DIR/wait_for.sh" http://127.0.0.1:$LLM_PORT/health   || exit 1
export HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" HF_HUB_OFFLINE="$HF_OFFLINE" TRANSFORMERS_OFFLINE="$HF_OFFLINE"
export PARSER_BACKEND=ollama                       # = self-hosted GLM-OCR (vLLM here)
export CONFIG_YAML_PATH="$CONF_DIR/glmocr.vllm.yaml"
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_COLLECTION_NAME="\${QDRANT_COLLECTION_NAME:-documents}"
export EMBEDDING_PROVIDER=qwen
export QWEN_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
export EMBEDDING_DIMENSIONS=2048
export RERANKER_BACKEND=qwen
export OPENAI_API_KEY="$VLLM_API_KEY"
export OPENAI_BASE_URL=http://127.0.0.1:$LLM_PORT/v1
export OPENAI_LLM_MODEL="$LLM_SERVED_NAME"
export IMAGE_CAPTION_ENABLED=true
export API_HOST=0.0.0.0 API_PORT=$APP_PORT
export API_WORKERS=1                               # >1 = duplicate models in VRAM
export PYTHONUNBUFFERED=1
cd "$APP_DIR" && exec "$VENV_APP/bin/python" scripts/serve.py --port $APP_PORT
SH
chmod +x "$RUN_DIR"/*.sh

# ── 8. supervisord ───────────────────────────────────────────────────────────
cat > "$SUPERVISOR_CONF" <<CONF
[unix_http_server]
file=$RUN_DIR/supervisor.sock

[supervisord]
logfile=$LOG_DIR/supervisord.log
pidfile=$RUN_DIR/supervisord.pid
childlogdir=$LOG_DIR

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://$RUN_DIR/supervisor.sock

[program:qdrant]
command=$RUN_DIR/qdrant.sh
priority=10
autorestart=true
stdout_logfile=$LOG_DIR/qdrant.log
redirect_stderr=true

[program:vllm-ocr]
command=$RUN_DIR/vllm-ocr.sh
priority=20
autorestart=true
startsecs=30
stopasgroup=true
killasgroup=true
stdout_logfile=$LOG_DIR/vllm-ocr.log
redirect_stderr=true

[program:vllm-llm]
command=$RUN_DIR/vllm-llm.sh
priority=30
autorestart=true
startsecs=30
stopasgroup=true
killasgroup=true
stdout_logfile=$LOG_DIR/vllm-llm.log
redirect_stderr=true

[program:app]
command=$RUN_DIR/app.sh
priority=40
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
stdout_logfile=$LOG_DIR/app.log
redirect_stderr=true
CONF

if [[ -S "$RUN_DIR/supervisor.sock" ]] && supervisorctl -c "$SUPERVISOR_CONF" status >/dev/null 2>&1; then
  log "supervisord already running — restarting services"
  supervisorctl -c "$SUPERVISOR_CONF" reread >/dev/null
  supervisorctl -c "$SUPERVISOR_CONF" update >/dev/null
  supervisorctl -c "$SUPERVISOR_CONF" restart all
else
  log "Starting supervisord..."
  supervisord -c "$SUPERVISOR_CONF"
fi

cat <<EOF

$(log "Launched. Models are cached; services need ~5–10 min to load + compile.")
  Status : bash $0 status        (supervisor state + endpoint health)
  Logs   : bash $0 logs          (or tail -F $LOG_DIR/vllm-ocr.log)
  Ready  : curl -s http://127.0.0.1:$APP_PORT/health
  GPU    : watch -n 2 nvidia-smi  (expect ~40–44 GB used when all are up)
EOF