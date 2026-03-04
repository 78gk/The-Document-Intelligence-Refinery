from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any
from enum import Enum

class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"

class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"

class DomainHint(str, Enum):
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    REGULATORY = "regulatory"
    GENERAL = "general"
    OTHER = "other"

class CostTier(str, Enum):
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"

class DocumentProfile(BaseModel):
    doc_id: str
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: str
    language_confidence: float = Field(ge=0, le=1)
    domain_hint: DomainHint
    domain_label: str = Field(default="general", min_length=1)
    extraction_cost_estimate: CostTier
    triage_confidence: float = Field(default=1.0, ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_domain_label(self) -> "DocumentProfile":
        if self.domain_hint != DomainHint.OTHER:
            self.domain_label = self.domain_hint.value
        return self
