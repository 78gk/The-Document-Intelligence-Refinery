from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    page: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x1 <= self.x0:
            raise ValueError("x1 must be greater than x0")
        if self.y1 <= self.y0:
            raise ValueError("y1 must be greater than y0")
        return self


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    PAGE_DUMP = "page_dump"
    TABLE = "table"
    FIGURE = "figure"
    VLM_OUTPUT = "vlm_output"


class ExtractionStrategyType(str, Enum):
    FAST_TEXT = "fast_text"
    LAYOUT_AWARE = "layout_aware"
    VISION = "vision"


class TextBlock(BaseModel):
    text: str = Field(min_length=1)
    bbox: BoundingBox
    block_type: BlockType
    reading_order: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class TableCell(BaseModel):
    text: str
    bbox: BoundingBox
    row_index: int = Field(ge=0)
    col_index: int = Field(ge=0)
    is_header: bool = False


class TableBlock(BaseModel):
    caption: Optional[str] = None
    rows: List[List[TableCell]]
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)


class FigureBlock(BaseModel):
    caption: Optional[str] = None
    bbox: BoundingBox
    image_path: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0, le=1)


class ExtractedDocument(BaseModel):
    doc_id: str = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[TableBlock] = Field(default_factory=list)
    figures: List[FigureBlock] = Field(default_factory=list)
    reading_order: List[int] = Field(default_factory=list)
    extraction_strategy: ExtractionStrategyType
    confidence_score: float = Field(ge=0, le=1)
    processing_time: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_reading_order(self) -> "ExtractedDocument":
        if not self.text_blocks:
            self.reading_order = []
            return self

        block_orders = [block.reading_order for block in self.text_blocks]
        normalized_orders = sorted(set(block_orders))

        if len(normalized_orders) != len(block_orders):
            raise ValueError("TextBlock reading_order values must be unique per document")

        if self.reading_order:
            if sorted(self.reading_order) != normalized_orders:
                raise ValueError("reading_order must match TextBlock reading_order values")
        else:
            self.reading_order = normalized_orders

        return self
