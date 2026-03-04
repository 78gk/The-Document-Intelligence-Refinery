from src.agents.extractor import ExtractionRouter
from src.models.document import ExtractedDocument, ExtractionStrategyType
from src.models.profile import CostTier, DocumentProfile, DomainHint, LayoutComplexity, OriginType


class DummyExtractor:
    def __init__(self, strategy: ExtractionStrategyType, confidence: float):
        self.strategy = strategy
        self.confidence = confidence

    def extract(self, doc_path: str, pages=None) -> ExtractedDocument:
        return ExtractedDocument(
            doc_id="doc",
            extraction_strategy=self.strategy,
            confidence_score=self.confidence,
            processing_time=0.01,
            metadata={},
        )

    def get_confidence_score(self, extracted_doc, doc_path=None, signals=None):
        return self.confidence


def _config():
    return {
        "extraction": {
            "graceful_degradation_threshold": 0.8,
            "escalation_rules": [
                {"from": "fast_text", "to": "layout_aware"},
                {"from": "layout_aware", "to": "vision"},
            ],
            "fast_text": {"min_confidence_threshold": 0.8},
            "layout_aware": {"min_confidence_threshold": 0.7},
            "vision": {"min_confidence_threshold": 0.6},
        }
    }


def _profile():
    return DocumentProfile(
        doc_id="doc",
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        language="en",
        language_confidence=1.0,
        domain_hint=DomainHint.GENERAL,
        extraction_cost_estimate=CostTier.FAST_TEXT_SUFFICIENT,
        triage_confidence=0.9,
    )


def test_router_escalates_a_to_b_to_c_and_sets_trace():
    router = ExtractionRouter(_config())
    router.fast_extractor = DummyExtractor(ExtractionStrategyType.FAST_TEXT, 0.2)
    router.layout_extractor = DummyExtractor(ExtractionStrategyType.LAYOUT_AWARE, 0.3)
    router.vision_extractor = DummyExtractor(ExtractionStrategyType.VISION, 0.4)
    router.strategy_map = {
        "fast_text": router.fast_extractor,
        "layout_aware": router.layout_extractor,
        "vision": router.vision_extractor,
    }

    result = router.extract("fake.pdf", _profile())

    assert result.extraction_strategy == ExtractionStrategyType.VISION
    assert result.metadata["review_required"] is True
    assert len(result.metadata["routing_trace"]) == 3
