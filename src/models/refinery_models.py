from enum import Enum
from typing import Any, Dict, List, Optional, Self

from pydantic import BaseModel, Field, model_validator

from .document import BoundingBox


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"


class ChunkRelationType(str, Enum):
    PARENT = "parent"
    CHILD = "child"
    CONTINUATION = "continuation"
    SIBLING = "sibling"
    REFERENCES = "references"


class ChunkRelationship(BaseModel):
    relation_type: ChunkRelationType
    target_chunk_id: str = Field(min_length=1)
    score: float = Field(default=1.0, ge=0, le=1)


class Provenance(BaseModel):
    document_name: str = Field(min_length=1)
    page_number: int = Field(gt=0)
    bbox: BoundingBox
    content_hash: str = Field(min_length=8)


class LDU(BaseModel):
    ldu_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    chunk_type: ChunkType
    page_refs: List[int]
    bounding_box: BoundingBox
    parent_section: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = Field(default_factory=list)
    token_count: int = Field(gt=0)
    content_hash: str = Field(min_length=8)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[ChunkRelationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        if not self.page_refs:
            raise ValueError("page_refs must include at least one page number")
        if any(page <= 0 for page in self.page_refs):
            raise ValueError("page_refs must contain positive page numbers")
        self.page_refs = sorted(set(self.page_refs))
        return self


class SectionNode(BaseModel):
    node_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    level: int = Field(ge=0)
    child_sections: List["SectionNode"] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    data_types_present: List[ChunkType] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError(
                f"page_end ({self.page_end}) cannot be less than page_start ({self.page_start})"
            )
        return self


class PageIndex(BaseModel):
    doc_id: str = Field(min_length=1)
    root: SectionNode


class ProvenanceChain(BaseModel):
    document_name: str = Field(min_length=1)
    citations: List[Provenance] = Field(min_length=1)
