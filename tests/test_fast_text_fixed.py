import pytest
from src.strategies.fast_text import FastTextExtractor
from src.models.document import BlockType
from unittest.mock import MagicMock

def test_fast_text_spatial_provenance():
    config = {
        "extraction": {
            "fast_text": {
                "min_char_count_per_page": 100,
                "min_char_density": 0.0001,
                "max_image_area_ratio": 0.5,
                "min_confidence_threshold": 0.8,
                "confidence_weights": {
                    "char_count": 0.25,
                    "char_density": 0.35,
                    "font_coverage": 0.25,
                    "image_quality": 0.15,
                },
            }
        }
    }
    extractor = FastTextExtractor(config)
    
    # Mock pdfplumber Table with actual coordinates
    mock_table = MagicMock()
    mock_table.bbox = [50, 50, 200, 100]
    mock_table.extract.return_value = [["Header", "Value"], ["Net Profit", "100M"]]
    
    # Mock cell coordinates
    mock_table.cells = [
        [[50, 50, 125, 75], [125, 50, 200, 75]],
        [[50, 75, 125, 100], [125, 75, 200, 100]]
    ]
    
    # Mock PDF Page
    mock_page = MagicMock()
    mock_page.width = 1000
    mock_page.height = 1000
    mock_page.page_number = 1
    mock_page.extract_text.return_value = "Financial Report"
    mock_page.find_tables.return_value = [mock_table]
    mock_page.chars = [{"text": "a", "fontname": "Arial", "width": 10, "x0": 0}] * 10
    mock_page.images = []
    
    # Mock PDF object
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    
    # Monkeypatch pdfplumber.open
    import pdfplumber
    original_open = pdfplumber.open
    pdfplumber.open = MagicMock(return_value=mock_pdf)
    mock_pdf.__enter__.return_value = mock_pdf

    try:
        doc = extractor.extract("fake_path.pdf")
        
        # Verify Table cell bboxes are NOT zero-area
        table = doc.tables[0]
        cell = table.rows[0][0]
        assert cell.bbox.x0 == 50
        assert cell.bbox.y0 == 50
        assert cell.bbox.x1 == 125
        assert cell.bbox.y1 == 75
        assert cell.bbox.page == 1
        
        # Verify BlockType Enum
        assert doc.text_blocks[0].block_type == BlockType.PAGE_DUMP
        assert doc.text_blocks[0].reading_order == 0
        
    finally:
        pdfplumber.open = original_open

def test_fast_text_confidence_signals():
    config = {
        "extraction": {
            "fast_text": {
                "min_confidence_threshold": 0.8,
                "min_char_count_per_page": 10,
                "min_char_density": 0.000005,
                "confidence_weights": {
                    "char_count": 0.25,
                    "char_density": 0.35,
                    "font_coverage": 0.25,
                    "image_quality": 0.15,
                },
            }
        }
    }
    extractor = FastTextExtractor(config)

    try:
        confidence = extractor.get_confidence_score(
            None,
            signals={
                "char_count": 1000,
                "char_density": 0.001,
                "font_coverage": 1.0,
                "image_area_ratio": 0.25,
            },
        )
        assert confidence > 0.9
    except Exception as e:
        pytest.fail(f"confidence scoring should not fail: {e}")
