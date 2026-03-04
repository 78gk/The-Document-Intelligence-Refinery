from src.strategies.layout_aware import LayoutAwareExtractor


def test_layout_aware_confidence_is_computed_from_extracted_content():
    config = {
        "extraction": {
            "layout_aware": {
                "tool_name": "Docling",
                "min_confidence_threshold": 0.7,
                "target_text_blocks": 1,
                "target_structured_blocks": 1,
                "confidence_weights": {
                    "block_quality": 0.5,
                    "structure_quality": 0.5,
                },
            }
        }
    }
    extractor = LayoutAwareExtractor(config)

    extractor._collect_layout_payload = lambda *_args, **_kwargs: {
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {
                        "type": "text",
                        "text": "Heading content",
                        "bbox": [0.0, 0.0, 200.0, 300.0],
                        "page": 1,
                        "confidence": 0.9,
                        "reading_order": 0,
                    },
                    {
                        "type": "table",
                        "bbox": [10.0, 50.0, 190.0, 120.0],
                        "page": 1,
                        "confidence": 0.95,
                        "caption": "T1",
                        "rows": [["H1", "H2"], ["v1", "v2"]],
                    },
                ],
            }
        ]
    }

    doc = extractor.extract("fake.pdf")
    assert doc.confidence_score > 0.0
    assert doc.metadata["tables"] == 1
