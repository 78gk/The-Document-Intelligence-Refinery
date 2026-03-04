import time
import pdfplumber
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import BaseExtractor
from ..models.document import ExtractedDocument, TextBlock, TableBlock, TableCell, FigureBlock, BoundingBox, BlockType


class LayoutPayloadAdapter:
    def collect_payload(self, doc_path: str, pages: Optional[list[int]]) -> Dict[str, Any]:
        raise NotImplementedError


class PdfPlumberLayoutAdapter(LayoutPayloadAdapter):
    def collect_payload(self, doc_path: str, pages: Optional[list[int]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"pages": []}
        with pdfplumber.open(doc_path) as pdf:
            target_pages = [pdf.pages[i] for i in pages] if pages else pdf.pages
            for page in target_pages:
                blocks = []
                text = page.extract_text() or ""
                if text.strip():
                    blocks.append(
                        {
                            "type": "text",
                            "text": text,
                            "bbox": [0.0, 0.0, float(page.width), float(page.height)],
                            "page": page.page_number,
                            "confidence": 0.9,
                            "reading_order": 0,
                        }
                    )
                for table in page.find_tables():
                    table_rows = table.extract() or []
                    tb = list(table.bbox)
                    if len(tb) == 4:
                        tb = [max(0.0, float(tb[0])), max(0.0, float(tb[1])), max(0.0, float(tb[2])), max(0.0, float(tb[3]))]
                        if tb[2] <= tb[0]:
                            tb[2] = tb[0] + 1.0
                        if tb[3] <= tb[1]:
                            tb[3] = tb[1] + 1.0
                    blocks.append(
                        {
                            "type": "table",
                            "bbox": tb,
                            "page": page.page_number,
                            "confidence": 0.92,
                            "caption": None,
                            "rows": table_rows,
                        }
                    )
                for idx, image in enumerate(page.images):
                    x0 = float(image.get("x0", 0.0))
                    y0 = float(image.get("y0", 0.0))
                    x1 = float(image.get("x1", x0 + float(image.get("width", 1.0))))
                    y1 = float(image.get("y1", y0 + float(image.get("height", 1.0))))

                    # Some PDFs yield slightly negative coords; clamp to satisfy schema.
                    x0 = max(0.0, x0)
                    y0 = max(0.0, y0)
                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)

                    if x1 <= x0:
                        x1 = x0 + 1.0
                    if y1 <= y0:
                        y1 = y0 + 1.0
                    blocks.append(
                        {
                            "type": "figure",
                            "bbox": [x0, y0, x1, y1],
                            "page": page.page_number,
                            "caption": f"Detected figure {idx + 1}",
                            "confidence": 0.85,
                        }
                    )
                payload["pages"].append({"page_number": page.page_number, "blocks": blocks})
        return payload


class ExternalCommandLayoutAdapter(LayoutPayloadAdapter):
    def __init__(self, command_template: List[str], timeout_seconds: int = 90):
        self.command_template = command_template
        self.timeout_seconds = timeout_seconds

    def collect_payload(self, doc_path: str, pages: Optional[list[int]]) -> Dict[str, Any]:
        if not self.command_template:
            raise ValueError("layout_aware.external_command.command_template is required")

        pages_arg = ",".join(str(p) for p in (pages or []))
        command = [
            arg.replace("{doc_path}", doc_path).replace("{pages}", pages_arg)
            for arg in self.command_template
        ]
        proc = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        payload = json.loads(proc.stdout)
        if not isinstance(payload, dict) or "pages" not in payload:
            raise ValueError("external layout adapter must return a JSON object with a 'pages' key")
        return payload


class LayoutAwareExtractor(BaseExtractor):
    """
    Simulates a layout-aware extractor (e.g., MinerU or Docling).
    Maps simulated external structure to internal ExtractedDocument schema.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cfg = self.config.get("extraction", {}).get("layout_aware", {})
        adapter_mode = str(cfg.get("adapter_mode", "pdfplumber")).lower()
        self.adapter_name = "pdfplumber"
        self.adapter: LayoutPayloadAdapter = PdfPlumberLayoutAdapter()
        if adapter_mode == "external_command":
            external_cfg = cfg.get("external_command", {})
            self.adapter = ExternalCommandLayoutAdapter(
                command_template=list(external_cfg.get("command_template", [])),
                timeout_seconds=int(external_cfg.get("timeout_seconds", 90)),
            )
            self.adapter_name = "external_command"

    def extract(self, doc_path: str, pages: list[int] = None) -> ExtractedDocument:
        start_time = time.time()
        doc_id = Path(doc_path).stem

        try:
            external_payload = self._collect_layout_payload(doc_path, pages)
            text_blocks, tables, figures = self._adapt_external_payload(doc_id, external_payload)
        except Exception as e:
            return self._empty_doc(doc_id, start_time, str(e))

        end_time = time.time()
        pages_processed = len(external_payload.get('pages', [])) if isinstance(external_payload, dict) else 0
        estimated_cost_usd = float(self.config.get('extraction', {}).get('layout_aware', {}).get('estimated_cost_per_page', 0.0)) * max(1, pages_processed)

        doc = ExtractedDocument(
            doc_id=doc_id,
            metadata={
                "strategy": "LayoutAwareExtractor",
                "tool": self.config.get("extraction", {}).get("layout_aware", {}).get("tool_name", "Docling"),
                "adapter": self.adapter_name,
                "reading_order_validated": True,
                "text_blocks": len(text_blocks),
                "tables": len(tables),
                "figures": len(figures),
                "pages_processed": pages_processed,
                "estimated_cost_usd": estimated_cost_usd,
            },
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            extraction_strategy="layout_aware",
            confidence_score=0.0,
            processing_time=end_time - start_time
        )
        doc.confidence_score = self.get_confidence_score(doc)
        return doc

    def get_confidence_score(
        self,
        extracted_doc: Optional[ExtractedDocument],
        doc_path: Optional[str] = None,
        signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not extracted_doc:
            return 0.0
        cfg = self.config.get("extraction", {}).get("layout_aware", {})
        weights = cfg.get("confidence_weights", {})
        block_weight = float(weights.get("block_quality", 0.0))
        structure_weight = float(weights.get("structure_quality", 0.0))

        block_count = len(extracted_doc.text_blocks)
        table_count = len(extracted_doc.tables)
        figure_count = len(extracted_doc.figures)
        structured_count = table_count + figure_count

        target_text_blocks = int(cfg.get("target_text_blocks", 1))
        target_structured_blocks = int(cfg.get("target_structured_blocks", 1))
        block_quality = float(min(1.0, block_count / max(target_text_blocks, 1)))
        structure_quality = float(min(1.0, structured_count / max(target_structured_blocks, 1)))
        extraction_confidences = [b.confidence for b in extracted_doc.text_blocks]
        extraction_confidences.extend([t.confidence for t in extracted_doc.tables])
        extraction_confidences.extend([f.confidence for f in extracted_doc.figures])
        model_confidence = (
            float(sum(extraction_confidences) / len(extraction_confidences))
            if extraction_confidences
            else 0.0
        )
        orders = [block.reading_order for block in extracted_doc.text_blocks]
        if not orders:
            reading_order_quality = 1.0
        else:
            unique_orders = sorted(set(orders))
            expected = list(range(min(unique_orders), min(unique_orders) + len(unique_orders)))
            reading_order_quality = 1.0 if unique_orders == expected else 0.5

        model_weight = float(weights.get("model_confidence", 0.2))
        ordering_weight = float(weights.get("reading_order", 0.1))
        score = (
            (block_weight * block_quality)
            + (structure_weight * structure_quality)
            + (model_weight * model_confidence)
            + (ordering_weight * reading_order_quality)
        )
        return float(max(0.0, min(1.0, score)))

    def _collect_layout_payload(self, doc_path: str, pages: Optional[list[int]]) -> Dict[str, Any]:
        return self.adapter.collect_payload(doc_path, pages)

    def _adapt_external_payload(
        self, doc_id: str, payload: Dict[str, Any]
    ) -> tuple[list[TextBlock], list[TableBlock], list[FigureBlock]]:
        text_blocks: List[TextBlock] = []
        tables: List[TableBlock] = []
        figures: List[FigureBlock] = []

        global_reading_order = 0
        for page_payload in payload.get("pages", []):
            page_number = page_payload["page_number"]
            blocks = sorted(page_payload.get("blocks", []), key=lambda b: b.get("reading_order", 0))
            for order, block in enumerate(blocks):
                bbox = block.get("bbox", [0.0, 0.0, 1.0, 1.0])
                x0, y0, x1, y1 = [float(v) for v in bbox]

                # Clamp to satisfy schema constraints (some PDFs can produce negative coords).
                x0 = max(0.0, x0)
                y0 = max(0.0, y0)
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)

                if x1 <= x0:
                    x1 = x0 + 1.0
                if y1 <= y0:
                    y1 = y0 + 1.0
                shared_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page_number)

                if block.get("type") == "text":
                    text = (block.get("text") or "").strip()
                    if text:
                        text_blocks.append(
                            TextBlock(
                                text=text,
                                bbox=shared_bbox,
                                block_type=BlockType.PARAGRAPH if order else BlockType.HEADING,
                                reading_order=global_reading_order,
                                confidence=float(block.get("confidence", 0.9)),
                            )
                        )
                        global_reading_order += 1
                elif block.get("type") == "table":
                    row_models: List[List[TableCell]] = []
                    for row_idx, row in enumerate(block.get("rows", [])):
                        cell_models = []
                        row = row or []
                        col_width = (x1 - x0) / max(1, len(row))
                        for col_idx, cell in enumerate(row):
                            cell_x0 = x0 + (col_idx * col_width)
                            cell_x1 = min(x1, cell_x0 + col_width)
                            cell_y0 = y0 + (row_idx * 10.0)
                            cell_y1 = min(y1, cell_y0 + 10.0)
                            if cell_x1 <= cell_x0:
                                cell_x1 = cell_x0 + 1.0
                            if cell_y1 <= cell_y0:
                                cell_y1 = cell_y0 + 1.0
                            cell_models.append(
                                TableCell(
                                    text=str(cell or ""),
                                    bbox=BoundingBox(x0=cell_x0, y0=cell_y0, x1=cell_x1, y1=cell_y1, page=page_number),
                                    row_index=row_idx,
                                    col_index=col_idx,
                                    is_header=(row_idx == 0),
                                )
                            )
                        row_models.append(cell_models)
                    tables.append(
                        TableBlock(
                            caption=block.get("caption"),
                            rows=row_models,
                            bbox=shared_bbox,
                            confidence=float(block.get("confidence", 0.9)),
                        )
                    )
                elif block.get("type") == "figure":
                    figures.append(
                        FigureBlock(
                            caption=block.get("caption"),
                            bbox=shared_bbox,
                            image_path=f"data/figures/{doc_id}_p{page_number}.png",
                            confidence=float(block.get("confidence", 0.8)),
                        )
                    )
        return text_blocks, tables, figures

    def _empty_doc(self, doc_id: str, start_time: float, error: str) -> ExtractedDocument:
        return ExtractedDocument(
            doc_id=doc_id,
            metadata={"error": "adapter_failure", "reason": error},
            extraction_strategy="layout_aware",
            confidence_score=0.0,
            processing_time=time.time() - start_time
        )
