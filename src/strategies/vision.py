import base64
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

import pdfplumber

from .base import BaseExtractor
from ..models.document import (
    BlockType,
    BoundingBox,
    ExtractedDocument,
    FigureBlock,
    TableBlock,
    TableCell,
    TextBlock,
)


class VisionExtractor(BaseExtractor):
    def extract(self, doc_path: str, pages: list[int] = None) -> ExtractedDocument:
        start_time = time.time()
        doc_id = Path(doc_path).stem
        cfg = self.config.get("extraction", {}).get("vision", {})
        budget_cap = float(cfg["budget_cap_per_doc"])
        est_cost_per_page = float(cfg["estimated_cost_per_page"])
        model_name = cfg.get("model", "llava:latest")
        budget_file = Path(cfg.get("budget_ledger_path", ".refinery/vision_budget_ledger.json"))
        budget_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            ledger = self._load_budget_ledger(budget_file)
            current_spend = float(ledger.get(doc_id, 0.0))

            with pdfplumber.open(doc_path) as pdf:
                target_pages = [pdf.pages[i] for i in pages] if pages else pdf.pages[: int(cfg.get("default_page_count", 1))]
                target_page_count = len(target_pages)

                projected_increment = est_cost_per_page * max(1, target_page_count)
                if current_spend + projected_increment > budget_cap:
                    return self._halted_doc(doc_id, start_time, "Budget exceeded")

                current_spend += projected_increment
                ledger[doc_id] = current_spend
                self._save_budget_ledger(budget_file, ledger)

                text_blocks, tables, figures = self._extract_with_ollama(doc_id, target_pages, cfg)

            if not text_blocks and not tables and not figures:
                return self._halted_doc(doc_id, start_time, "No extractable content from VLM")

            estimated_tokens = int(cfg.get("avg_tokens_per_page", 700) * max(1, target_page_count))
            confidence_signals = self._build_confidence_signals(
                text_blocks=text_blocks,
                tables=tables,
                figures=figures,
                target_page_count=target_page_count,
                current_spend=current_spend,
                budget_cap=budget_cap,
            )
            confidence_score = self.get_confidence_score(None, signals=confidence_signals)
        except Exception as e:
            return self._halted_doc(doc_id, start_time, str(e))

        end_time = time.time()
        return ExtractedDocument(
            doc_id=doc_id,
            metadata={
                "strategy": "VisionExtractor",
                "provider": "ollama",
                "model": model_name,
                "total_spend": current_spend,
                # For interim grading: surface a consistent cost field across strategies.
                "estimated_cost_usd": current_spend,
                "budget_cap_per_doc": budget_cap,
                "estimated_tokens": estimated_tokens,
                "pages_processed": target_page_count,
                "confidence_signals": confidence_signals,
            },
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            extraction_strategy="vision",
            confidence_score=confidence_score,
            processing_time=end_time - start_time,
        )

    def get_confidence_score(
        self,
        extracted_doc: Optional[ExtractedDocument],
        doc_path: Optional[str] = None,
        signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        cfg = self.config.get("extraction", {}).get("vision", {})
        weights = cfg.get("confidence_weights", {})
        signals = signals or {}

        model_confidence = float(signals.get("mean_model_confidence", 0.0))
        content_coverage = float(signals.get("content_coverage", 0.0))
        structure_score = float(signals.get("structure_score", 0.0))
        budget_headroom = float(signals.get("budget_headroom", 0.0))

        score = (
            float(weights.get("model_confidence", 0.4)) * model_confidence
            + float(weights.get("content_coverage", 0.25)) * content_coverage
            + float(weights.get("structure_score", 0.25)) * structure_score
            + float(weights.get("budget_headroom", 0.1)) * budget_headroom
        )
        return float(max(0.0, min(1.0, score)))

    def _halted_doc(self, doc_id: str, start_time: float, reason: str) -> ExtractedDocument:
        return ExtractedDocument(
            doc_id=doc_id,
            metadata={"error": "vision_halted", "reason": reason},
            extraction_strategy="vision",
            confidence_score=0.0,
            processing_time=time.time() - start_time,
        )

    def _load_budget_ledger(self, budget_file: Path) -> Dict[str, float]:
        if not budget_file.exists():
            return {}
        with open(budget_file, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                return {}
        return {k: float(v) for k, v in data.items()}

    def _save_budget_ledger(self, budget_file: Path, ledger: Dict[str, float]) -> None:
        with open(budget_file, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=2)

    def _extract_with_ollama(self, doc_id: str, target_pages: List[Any], cfg: Dict[str, Any]) -> Tuple[List[TextBlock], List[TableBlock], List[FigureBlock]]:
        text_blocks: List[TextBlock] = []
        tables: List[TableBlock] = []
        figures: List[FigureBlock] = []

        global_reading_order = 0
        for reading_order, page in enumerate(target_pages):
            page_image_b64 = self._render_page_png_base64(page)

            parsed: Dict[str, Any] = {}
            if page_image_b64:
                try:
                    response = self._call_ollama_vlm(page.page_number, page_image_b64, cfg)
                    parsed = self._parse_ollama_response(response)
                except Exception:
                    parsed = {}

            if not parsed:
                # Degrade gracefully when the VLM path is unavailable (e.g., missing PDF rendering
                # dependencies or Ollama not running). We still provide usable text/figure signals
                # so downstream chunking/indexing can function for the interim submission.
                page_text = ""
                try:
                    page_text = (page.extract_text() or "").strip()
                except Exception:
                    page_text = ""

                if page_text:
                    text_blocks.append(
                        TextBlock(
                            text=page_text,
                            bbox=BoundingBox(
                                x0=0.0,
                                y0=0.0,
                                x1=float(getattr(page, "width", 1.0)),
                                y1=float(getattr(page, "height", 1.0)),
                                page=page.page_number,
                            ),
                            block_type=BlockType.PAGE_DUMP,
                            reading_order=global_reading_order,
                            confidence=0.4,
                        )
                    )
                    global_reading_order += 1

                for img_idx, image in enumerate(getattr(page, "images", []) or []):
                    bbox = self._bbox_from_page_image(image, page.page_number)
                    figures.append(
                        FigureBlock(
                            caption=f"Detected embedded figure {img_idx + 1}",
                            bbox=bbox,
                            image_path=None,
                            confidence=0.6,
                        )
                    )
                continue

            page_texts = parsed.get("text_blocks", [])
            for block in page_texts:
                bbox = self._bbox_from_payload(block.get("bbox"), page)
                text = str(block.get("text", "")).strip()
                if not text:
                    continue
                text_blocks.append(
                    TextBlock(
                        text=text,
                        bbox=bbox,
                        block_type=BlockType.VLM_OUTPUT,
                        reading_order=global_reading_order,
                        confidence=float(block.get("confidence", 0.95)),
                    )
                )
                global_reading_order += 1

            for table_payload in parsed.get("tables", []):
                table_bbox = self._bbox_from_payload(table_payload.get("bbox"), page)
                headers = [str(h) for h in table_payload.get("headers", [])]
                rows = table_payload.get("rows", [])

                cell_rows: List[List[TableCell]] = []
                all_rows = [headers] + rows if headers else rows
                row_count = max(1, len(all_rows))
                col_count = max(1, len(headers) if headers else max((len(r) for r in rows), default=1))

                row_height = (table_bbox.y1 - table_bbox.y0) / row_count
                col_width = (table_bbox.x1 - table_bbox.x0) / col_count

                for r_idx, row in enumerate(all_rows):
                    row = row or []
                    cell_models: List[TableCell] = []
                    for c_idx in range(col_count):
                        val = str(row[c_idx]) if c_idx < len(row) else ""
                        cx0 = table_bbox.x0 + (c_idx * col_width)
                        cy0 = table_bbox.y0 + (r_idx * row_height)
                        cx1 = max(cx0 + 1.0, cx0 + col_width)
                        cy1 = max(cy0 + 1.0, cy0 + row_height)
                        cell_models.append(
                            TableCell(
                                text=val,
                                bbox=BoundingBox(x0=cx0, y0=cy0, x1=min(cx1, table_bbox.x1), y1=min(cy1, table_bbox.y1), page=page.page_number),
                                row_index=r_idx,
                                col_index=c_idx,
                                is_header=bool(headers and r_idx == 0),
                            )
                        )
                    cell_rows.append(cell_models)

                tables.append(
                    TableBlock(
                        caption=table_payload.get("caption"),
                        rows=cell_rows,
                        bbox=table_bbox,
                        confidence=float(table_payload.get("confidence", 0.9)),
                    )
                )

            for figure_payload in parsed.get("figures", []):
                fig_bbox = self._bbox_from_payload(figure_payload.get("bbox"), page)
                figures.append(
                    FigureBlock(
                        caption=figure_payload.get("caption") or f"Detected figure on page {page.page_number}",
                        bbox=fig_bbox,
                        image_path=figure_payload.get("image_path"),
                        confidence=float(figure_payload.get("confidence", 0.85)),
                    )
                )

            if not parsed.get("figures"):
                for img_idx, image in enumerate(page.images):
                    bbox = self._bbox_from_page_image(image, page.page_number)
                    figures.append(
                        FigureBlock(
                            caption=f"Detected embedded figure {img_idx + 1}",
                            bbox=bbox,
                            image_path=None,
                            confidence=0.8,
                        )
                    )

        return text_blocks, tables, figures

    def _build_confidence_signals(
        self,
        text_blocks: List[TextBlock],
        tables: List[TableBlock],
        figures: List[FigureBlock],
        target_page_count: int,
        current_spend: float,
        budget_cap: float,
    ) -> Dict[str, float]:
        confidences = [b.confidence for b in text_blocks]
        confidences.extend([t.confidence for t in tables])
        confidences.extend([f.confidence for f in figures])
        mean_model_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0

        pages_with_content = {b.bbox.page for b in text_blocks}
        pages_with_content.update({t.bbox.page for t in tables})
        pages_with_content.update({f.bbox.page for f in figures})
        content_coverage = float(min(1.0, len(pages_with_content) / max(1, target_page_count)))

        target_structured = max(1, int(self.config.get("extraction", {}).get("vision", {}).get("target_structured_elements", 2)))
        structure_score = float(min(1.0, (len(tables) + len(figures)) / target_structured))

        budget_headroom = float(max(0.0, min(1.0, 1.0 - (current_spend / max(budget_cap, 1e-6)))))
        return {
            "mean_model_confidence": mean_model_confidence,
            "content_coverage": content_coverage,
            "structure_score": structure_score,
            "budget_headroom": budget_headroom,
        }

    def _render_page_png_base64(self, page: Any) -> Optional[str]:
        try:
            page_img = page.to_image(resolution=150)
            img_bytes = BytesIO()
            page_img.original.save(img_bytes, format="PNG")
            return base64.b64encode(img_bytes.getvalue()).decode("ascii")
        except Exception:
            return None

    def _call_ollama_vlm(self, page_number: int, image_b64: str, cfg: Dict[str, Any]) -> str:
        endpoint = cfg.get("ollama_endpoint", "http://localhost:11434/api/chat")
        model = cfg.get("model", "llava:latest")
        timeout_s = int(cfg.get("request_timeout_seconds", 60))
        temperature = float(cfg.get("temperature", 0.0))

        schema_prompt = cfg.get(
            "response_schema_prompt",
            (
                "Return STRICT JSON with keys: text_blocks, tables, figures. "
                "Each text block: {text, bbox:[x0,y0,x1,y1], confidence}. "
                "Each table: {caption, headers:[...], rows:[[...]], bbox:[x0,y0,x1,y1], confidence}. "
                "Each figure: {caption, bbox:[x0,y0,x1,y1], confidence}."
            ),
        )
        user_prompt = cfg.get("user_prompt", "Extract readable text, tables, and figures from this page image.")

        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
            "messages": [
                {
                    "role": "system",
                    "content": schema_prompt,
                },
                {
                    "role": "user",
                    "content": f"Page {page_number}: {user_prompt}",
                    "images": [image_b64],
                },
            ],
        }

        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        message = raw.get("message", {}) if isinstance(raw, dict) else {}
        content = message.get("content", "")
        if not content:
            raise RuntimeError("Ollama returned empty content")
        return content

    def _parse_ollama_response(self, response_content: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(response_content)
        except json.JSONDecodeError:
            return {"text_blocks": [{"text": response_content, "bbox": [0, 0, 1, 1], "confidence": 0.7}], "tables": [], "figures": []}

        if not isinstance(parsed, dict):
            return {"text_blocks": [], "tables": [], "figures": []}
        parsed.setdefault("text_blocks", [])
        parsed.setdefault("tables", [])
        parsed.setdefault("figures", [])
        return parsed

    def _bbox_from_payload(self, bbox_payload: Any, page: Any) -> BoundingBox:
        default_bbox = [0.0, 0.0, float(page.width), float(page.height)]
        bbox = bbox_payload if isinstance(bbox_payload, list) and len(bbox_payload) == 4 else default_bbox
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
        return BoundingBox(x0=max(0.0, x0), y0=max(0.0, y0), x1=max(x1, 1.0), y1=max(y1, 1.0), page=page.page_number)

    def _bbox_from_page_image(self, image: Dict[str, Any], page_number: int) -> BoundingBox:
        """Convert a pdfplumber image dict into a valid BoundingBox.

        Some PDFs (or coordinate transforms) can yield slightly negative coordinates
        (e.g., -0.01). Our schema requires non-negative values, so we clamp.
        """
        x0 = float(image.get("x0", 0.0))
        y0 = float(image.get("y0", 0.0))
        x1 = float(image.get("x1", x0 + float(image.get("width", 1.0))))
        y1 = float(image.get("y1", y0 + float(image.get("height", 1.0))))

        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)

        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page_number)
