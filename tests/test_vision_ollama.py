import json
from unittest.mock import MagicMock

import pdfplumber

from src.strategies.vision import VisionExtractor


class _DummyPDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_vision_extractor_uses_ollama_and_returns_structured_outputs(tmp_path, monkeypatch):
    ledger_path = tmp_path / "vision_budget.json"
    config = {
        "extraction": {
            "vision": {
                "budget_cap_per_doc": 10.0,
                "estimated_cost_per_page": 0.1,
                "model": "llava:13b",
                "ollama_endpoint": "http://localhost:11434/api/chat",
                "request_timeout_seconds": 5,
                "avg_tokens_per_page": 100,
                "default_page_count": 1,
                "budget_ledger_path": str(ledger_path),
            }
        }
    }

    extractor = VisionExtractor(config)

    mock_page = MagicMock()
    mock_page.page_number = 1
    mock_page.width = 612
    mock_page.height = 792
    mock_page.images = [{"x0": 10, "y0": 20, "x1": 120, "y1": 200}]

    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _DummyPDF([mock_page]))
    monkeypatch.setattr(extractor, "_render_page_png_base64", lambda *_args, **_kwargs: "ZmFrZV9pbWFnZQ==")
    monkeypatch.setattr(
        extractor,
        "_call_ollama_vlm",
        lambda *_args, **_kwargs: json.dumps(
            {
                "text_blocks": [{"text": "OCR paragraph", "bbox": [0, 0, 300, 120], "confidence": 0.95}],
                "tables": [
                    {
                        "caption": "Revenue",
                        "headers": ["Year", "Amount"],
                        "rows": [["2024", "100"]],
                        "bbox": [50, 150, 400, 300],
                        "confidence": 0.9,
                    }
                ],
                "figures": [{"caption": "Balance chart", "bbox": [420, 120, 580, 360], "confidence": 0.88}],
            }
        ),
    )

    doc = extractor.extract("fake.pdf")

    assert doc.metadata["provider"] == "ollama"
    assert len(doc.text_blocks) == 1
    assert len(doc.tables) == 1
    assert len(doc.figures) == 1
    assert doc.tables[0].rows[0][0].is_header is True
    assert doc.confidence_score < 0.98
    assert "confidence_signals" in doc.metadata
