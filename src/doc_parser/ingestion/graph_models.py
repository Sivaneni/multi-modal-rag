"""Pydantic models for knowledge graph entities extracted from documents."""
from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str
    properties: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    properties: dict = Field(default_factory=dict)


class GraphComponents(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
