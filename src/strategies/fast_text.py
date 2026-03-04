import time
import pdfplumber
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base import BaseExtractor
from ..models.document import ExtractedDocument, TextBlock, TableBlock, TableCell, BoundingBox, BlockType

class FastTextExtractor(BaseExtractor):
    def extract(self, doc_path: str, pages: list[int] = None) -> ExtractedDocument:
        start_time = time.time()
        text_blocks = []
        tables = []
        doc_id = Path(doc_path).stem
        analysis_signals: Dict[str, float] = {}
        
        try:
            with pdfplumber.open(doc_path) as pdf:
                target_pages = [pdf.pages[i] for i in pages] if pages else pdf.pages
                total_chars = 0
                total_page_area = 0.0
                total_font_chars = 0
                total_image_area = 0.0
                reading_order = 0

                for page in target_pages:
                    page_area = float(page.width) * float(page.height)
                    total_page_area += page_area
                    page_chars = page.chars
                    total_chars += len(page_chars)
                    total_font_chars += len([c for c in page_chars if c.get("fontname")])
                    total_image_area += sum([float(img.get('width', 0)) * float(img.get('height', 0)) for img in page.images])

                    # Extract Text Blocks
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_blocks.append(TextBlock(
                            text=page_text,
                            bbox=BoundingBox(x0=0, y0=0, x1=float(page.width), y1=float(page.height), page=page.page_number),
                            block_type=BlockType.PAGE_DUMP,
                            reading_order=reading_order,
                            confidence=1.0
                        ))
                        reading_order += 1
                    
                    # Extract Tables
                    page_tables = page.find_tables()
                    for table in page_tables:
                        rows = []
                        table_data = table.extract()
                        for r_idx, row in enumerate(table_data):
                            cells = []
                            for c_idx, cell_text in enumerate(row):
                                if cell_text:
                                    cell_bbox = self._resolve_cell_bbox(table, r_idx, c_idx)
                                    cells.append(TableCell(
                                        text=cell_text,
                                        bbox=BoundingBox(x0=cell_bbox[0], y0=cell_bbox[1], x1=cell_bbox[2], y1=cell_bbox[3], page=page.page_number),
                                        row_index=r_idx,
                                        col_index=c_idx,
                                        is_header=(r_idx == 0)
                                    ))
                            rows.append(cells)
                        
                        if rows:
                            tables.append(TableBlock(
                                rows=rows,
                                bbox=BoundingBox(x0=table.bbox[0], y0=table.bbox[1], x1=table.bbox[2], y1=table.bbox[3], page=page.page_number),
                                confidence=0.9
                            ))
                analysis_signals = {
                    "char_count": float(total_chars),
                    "char_density": float(total_chars / total_page_area) if total_page_area > 0 else 0.0,
                    "font_coverage": float(total_font_chars / total_chars) if total_chars > 0 else 0.0,
                    "image_area_ratio": float(total_image_area / total_page_area) if total_page_area > 0 else 1.0,
                }
        except Exception as e:
            return ExtractedDocument(
                doc_id=doc_id,
                metadata={"strategy": "FastTextExtractor", "error": str(e)},
                extraction_strategy="fast_text",
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )

        end_time = time.time()
        doc = ExtractedDocument(
            doc_id=doc_id,
            metadata={"strategy": "FastTextExtractor", "signals": analysis_signals},
            text_blocks=text_blocks,
            tables=tables,
            extraction_strategy="fast_text",
            confidence_score=0.0,
            processing_time=end_time - start_time
        )
        doc.confidence_score = self.get_confidence_score(doc, doc_path, analysis_signals)
        return doc

    def get_confidence_score(
        self,
        extracted_doc: Optional[ExtractedDocument],
        doc_path: Optional[str] = None,
        signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        cfg = self.config.get("extraction", {}).get("fast_text", {})
        weights = cfg.get("confidence_weights", {})
        signals = signals or {}

        char_count = float(signals.get("char_count", 0.0))
        char_density = float(signals.get("char_density", 0.0))
        font_coverage = float(signals.get("font_coverage", 0.0))
        image_ratio = float(np.clip(signals.get("image_area_ratio", 1.0), 0.0, 1.0))

        if char_count <= 0:
            return 0.0

        min_char_count = float(cfg["min_char_count_per_page"])
        min_density = float(cfg["min_char_density"])

        normalized_count = float(np.clip(char_count / max(min_char_count, 1.0), 0.0, 1.0))
        normalized_density = float(np.clip(char_density / max(min_density, 1e-8), 0.0, 1.0))
        image_quality = float(np.clip(1.0 - image_ratio, 0.0, 1.0))

        confidence = (
            float(weights["char_count"]) * normalized_count
            + float(weights["char_density"]) * normalized_density
            + float(weights["font_coverage"]) * font_coverage
            + float(weights["image_quality"]) * image_quality
        )
        return float(np.clip(confidence, 0.0, 1.0))

    def _resolve_cell_bbox(self, table: Any, row_idx: int, col_idx: int) -> Tuple[float, float, float, float]:
        if hasattr(table, "cells") and table.cells:
            try:
                cell = table.cells[row_idx][col_idx]
                if cell and len(cell) == 4:
                    return float(cell[0]), float(cell[1]), float(cell[2]), float(cell[3])
            except Exception:
                pass
        return float(table.bbox[0]), float(table.bbox[1]), float(table.bbox[2]), float(table.bbox[3])
