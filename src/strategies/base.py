from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.document import ExtractedDocument

class BaseExtractor(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def extract(self, doc_path: str, pages: Optional[list[int]] = None) -> ExtractedDocument:
        """Extract content from the document."""
        pass

    @abstractmethod
    def get_confidence_score(
        self,
        extracted_doc: Optional[ExtractedDocument],
        doc_path: Optional[str] = None,
        signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate confidence score for the extraction."""
        pass
