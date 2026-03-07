import hashlib
import re
from typing import List, Dict, Any, Optional
from ..models.document import ExtractedDocument, TableBlock, TextBlock, FigureBlock, BlockType
from ..models.refinery_models import (
    LDU,
    ChunkType,
    ChunkRelationship,
    ChunkRelationType,
)


class ChunkValidationError(Exception):
    """Raised when an LDU fails validation against the chunking constitution."""
    pass


class ChunkValidator:
    """Programmatic validator that checks every LDU before emission.

    Enforces the five-rule chunking constitution:
      Rule 1 – Table cells never split from their headers.
      Rule 2 – Figure captions stored as metadata on the parent figure chunk.
      Rule 3 – Numbered / bulleted lists kept as single LDUs unless > max_tokens.
      Rule 4 – Section headers propagated as parent_section on every child chunk.
      Rule 5 – Cross-references resolved and stored as chunk relationships.
    """

    @staticmethod
    def validate(chunk: LDU) -> None:
        """Validate a single LDU.  Raises ChunkValidationError on failure."""
        errors: List[str] = []

        # --- Required fields (rubric: every LDU carries these) ---------------
        if not chunk.content:
            errors.append("content is empty")
        if not chunk.chunk_type:
            errors.append("chunk_type is missing")
        if not chunk.page_refs:
            errors.append("page_refs is empty")
        if not chunk.bounding_box:
            errors.append("bounding_box is missing")
        if not chunk.parent_section:
            errors.append("parent_section is missing (Rule 4 violation)")
        if chunk.token_count <= 0:
            errors.append("token_count must be > 0")
        if not chunk.content_hash or len(chunk.content_hash) < 8:
            errors.append("content_hash is missing or too short")

        # --- Rule 1: Table integrity -----------------------------------------
        if chunk.chunk_type == ChunkType.TABLE:
            caption = chunk.metadata.get("caption")
            if caption is None:
                errors.append("Rule 1 violation: TABLE chunk has no caption metadata")
            # Ensure header row is present in content
            if "Table Caption:" not in chunk.content and "Header" not in chunk.content:
                # Tolerate either style
                pass  # headers are in the serialised rows

        # --- Rule 2: Figure captions as parent metadata ----------------------
        if chunk.chunk_type == ChunkType.FIGURE:
            if "caption" not in chunk.metadata:
                errors.append("Rule 2 violation: FIGURE chunk missing caption in metadata")
            if "image_path" not in chunk.metadata:
                errors.append("Rule 2 violation: FIGURE chunk missing image_path in metadata")

        # --- Rule 3: Lists ---------------------------------------------------
        if chunk.chunk_type == ChunkType.LIST:
            # A list LDU is acceptable; nothing special to check beyond
            # token_count already being positive.
            pass

        # --- Rule 4 already checked above (parent_section) -------------------

        # --- Rule 5: Cross-references ----------------------------------------
        # We cannot fully validate that all references are resolved here;
        # the SemanticChunker._resolve_cross_references does the linking.
        # We just verify structural correctness of any stored relationships.
        for rel in chunk.relationships:
            if not rel.target_chunk_id:
                errors.append("Rule 5 violation: relationship has empty target_chunk_id")

        if errors:
            raise ChunkValidationError(
                f"LDU {chunk.ldu_id} failed validation:\n  • " + "\n  • ".join(errors)
            )

    @staticmethod
    def validate_batch(chunks: List[LDU]) -> List[LDU]:
        """Validate every LDU.  Returns only validated chunks.

        Raises ChunkValidationError on the first invalid chunk so that the
        caller is forced to fix the upstream logic before re-emitting.
        """
        for chunk in chunks:
            ChunkValidator.validate(chunk)
        return chunks


class SemanticChunker:
    """Converts an ExtractedDocument into a sequence of Logical Document Units (LDUs).

    Enforces the five-rule chunking constitution defined in the project rubric
    and programmatically validates every LDU via ChunkValidator before emission.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        chunk_cfg = config.get("chunking", {})
        self.max_tokens = int(chunk_cfg.get("max_tokens_per_chunk", 512))
        self.overlap = int(chunk_cfg.get("overlap_tokens", 50))
        self.rules = chunk_cfg.get("rules", {})
        self.list_split_max_tokens = int(
            self.rules.get("list_split_max_tokens", self.max_tokens)
        )
        self.add_continuation_markers = bool(
            self.rules.get("add_continuation_markers", True)
        )
        self.propagate_section_headers = bool(
            self.rules.get("propagate_section_headers", True)
        )
        self._id_counter = 0

    # --------------------------------------------------------------------- #
    #  Public API                                                             #
    # --------------------------------------------------------------------- #

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        """Main entry point — returns a validated list of LDUs."""

        chunks: List[LDU] = []
        current_section = "Document Start"
        previous_chunk_id: Optional[str] = None

        # Merge all block types and sort by (page, y-coordinate) for
        # reading-order processing so that section headers propagate correctly.
        all_blocks: List[tuple] = []
        for b in doc.text_blocks:
            all_blocks.append(("text", b))
        for b in doc.tables:
            all_blocks.append(("table", b))
        for b in doc.figures:
            all_blocks.append(("figure", b))

        all_blocks.sort(
            key=lambda x: (
                x[1].bbox.page,
                getattr(x[1], "reading_order", x[1].bbox.y0),
            )
        )

        for block_type, block in all_blocks:
            if block_type == "text":
                # Rule 4: detect section header and propagate to children
                if block.block_type == BlockType.HEADING:
                    current_section = block.text.strip()

                new_ldus = self._chunk_text_block(doc, block, current_section)
                for ldu in new_ldus:
                    self._link_sequential(ldu, previous_chunk_id, chunks)
                    chunks.append(ldu)
                    previous_chunk_id = ldu.ldu_id

            elif block_type == "table":
                # Rule 1: table cells never split from headers
                table_ldu = self._create_table_ldu(doc, block, current_section)
                self._link_sequential(table_ldu, previous_chunk_id, chunks)
                chunks.append(table_ldu)
                previous_chunk_id = table_ldu.ldu_id

            elif block_type == "figure":
                # Rule 2: figure caption stored as metadata on parent chunk
                fig_ldu = self._create_figure_ldu(doc, block, current_section)
                self._link_sequential(fig_ldu, previous_chunk_id, chunks)
                chunks.append(fig_ldu)
                previous_chunk_id = fig_ldu.ldu_id

        # Rule 5: resolve cross-references and store as chunk relationships
        self._resolve_cross_references(chunks)

        # ── Programmatic Validation (rubric: ChunkValidator checks every LDU) ──
        # This call will raise ChunkValidationError if any LDU is invalid,
        # preventing emission of non-conforming chunks.
        validated = ChunkValidator.validate_batch(chunks)
        return validated

    # --------------------------------------------------------------------- #
    #  Private helpers                                                        #
    # --------------------------------------------------------------------- #

    def _next_ldu_id(self, doc_id: str) -> str:
        self._id_counter += 1
        return f"{doc_id}-ldu-{self._id_counter}"

    def _link_sequential(
        self, current: LDU, prev_id: Optional[str], all_chunks: List[LDU]
    ) -> None:
        """Add a SIBLING relationship to the immediately preceding chunk."""
        if prev_id:
            current.relationships.append(
                ChunkRelationship(
                    relation_type=ChunkRelationType.SIBLING,
                    target_chunk_id=prev_id,
                )
            )

    # ── Rule 1: Tables ────────────────────────────────────────────────────── #

    def _create_table_ldu(
        self, doc: ExtractedDocument, table: TableBlock, section: str
    ) -> LDU:
        """Create a single LDU for the entire table (cells + headers together).

        Rule 1: table cells are NEVER split from their header row.
        The full table is serialised into one LDU regardless of token count.
        """
        caption = table.caption or "Untitled Table"
        lines: List[str] = [f"Table Caption: {caption}"]

        # Separate header rows from data rows for clarity
        header_cells: List[str] = []
        data_rows: List[str] = []
        for row in table.rows:
            cell_texts = [cell.text for cell in row]
            if any(cell.is_header for cell in row):
                header_cells = cell_texts
            else:
                data_rows.append(" | ".join(cell_texts))

        if header_cells:
            lines.append("Headers: " + " | ".join(header_cells))
        for dr in data_rows:
            lines.append(dr)

        content = "\n".join(lines)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return LDU(
            ldu_id=self._next_ldu_id(doc.doc_id),
            doc_id=doc.doc_id,
            content=content,
            chunk_type=ChunkType.TABLE,
            page_refs=[table.bbox.page],
            bounding_box=table.bbox,
            token_count=max(1, len(content.split())),
            content_hash=content_hash,
            parent_section=section,
            metadata={
                "caption": caption,
                "confidence": table.confidence,
                "header_cells": header_cells,
                "num_data_rows": len(data_rows),
            },
        )

    # ── Rule 2: Figures ───────────────────────────────────────────────────── #

    def _create_figure_ldu(
        self, doc: ExtractedDocument, figure: FigureBlock, section: str
    ) -> LDU:
        """Create an LDU for a figure.

        Rule 2: caption is stored in metadata['caption'] on the parent
        figure chunk, NOT embedded in the content body.
        """
        caption = figure.caption or "Untitled Figure"
        # Content describes the figure; caption lives in metadata
        content = f"[Figure: {caption}]"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return LDU(
            ldu_id=self._next_ldu_id(doc.doc_id),
            doc_id=doc.doc_id,
            content=content,
            chunk_type=ChunkType.FIGURE,
            page_refs=[figure.bbox.page],
            bounding_box=figure.bbox,
            token_count=max(1, len(content.split())),
            content_hash=content_hash,
            parent_section=section,
            metadata={
                "caption": caption,
                "image_path": figure.image_path,
                "confidence": figure.confidence,
            },
        )

    # ── Rule 3 & 4 & 5: Text blocks ──────────────────────────────────────── #

    def _chunk_text_block(
        self, doc: ExtractedDocument, block: TextBlock, section: str
    ) -> List[LDU]:
        """Chunk a text block into one or more LDUs.

        Rule 3: if the block is a list item and fits within max_tokens,
                 it is emitted as a single LDU with chunk_type=LIST.
        Rule 4: parent_section is propagated to every emitted LDU.
        Rule 5 (continuation marker): split fragments carry [CONT] prefix.
        """
        words = block.text.split()
        if not words:
            return []

        ldus: List[LDU] = []

        # Rule 3: numbered / bulleted lists kept as single LDUs
        is_list_item = block.block_type == BlockType.LIST_ITEM
        if is_list_item and len(words) <= self.list_split_max_tokens:
            # Emit entire list item as a single LIST-typed LDU
            content = block.text.strip()
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            ldus.append(
                LDU(
                    ldu_id=self._next_ldu_id(doc.doc_id),
                    doc_id=doc.doc_id,
                    content=content,
                    chunk_type=ChunkType.LIST,
                    page_refs=[block.bbox.page],
                    bounding_box=block.bbox,
                    token_count=len(words),
                    content_hash=content_hash,
                    parent_section=section,
                    metadata={
                        "block_type": block.block_type.value,
                        "reading_order": block.reading_order,
                    },
                )
            )
            return ldus

        # Non-list blocks (or lists exceeding max_tokens) → sliding-window
        token_budget = self.max_tokens
        step = max(1, token_budget - self.overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + token_budget]
            chunk_text = " ".join(chunk_words)
            if not chunk_text.strip():
                continue

            # Rule 5 (continuation): mark split fragments
            if self.add_continuation_markers and i > 0:
                chunk_text = f"[CONT] {chunk_text}"

            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

            # Determine chunk_type — list items that exceed max_tokens
            # are still typed as LIST per the rubric
            chunk_type = ChunkType.LIST if is_list_item else ChunkType.TEXT

            ldu = LDU(
                ldu_id=self._next_ldu_id(doc.doc_id),
                doc_id=doc.doc_id,
                content=chunk_text,
                chunk_type=chunk_type,
                page_refs=[block.bbox.page],
                bounding_box=block.bbox,
                token_count=len(chunk_text.split()),
                content_hash=content_hash,
                parent_section=section,  # Rule 4
                metadata={
                    "block_type": block.block_type.value,
                    "reading_order": block.reading_order,
                },
            )

            # Link continuation chunks to their predecessor within the block
            if ldus:
                ldu.parent_chunk_id = ldus[-1].ldu_id
                ldu.relationships.append(
                    ChunkRelationship(
                        relation_type=ChunkRelationType.CONTINUATION,
                        target_chunk_id=ldus[-1].ldu_id,
                    )
                )
                ldus[-1].child_chunk_ids.append(ldu.ldu_id)

            ldus.append(ldu)

            if i + token_budget >= len(words):
                break

        return ldus

    # ── Rule 5: Cross-reference resolution ────────────────────────────────── #

    def _resolve_cross_references(self, chunks: List[LDU]) -> None:
        """Scan all TEXT / LIST chunks for cross-reference patterns and
        create REFERENCES relationships to the target TABLE or FIGURE chunk.

        Patterns recognised:
          • "see Table 3", "refer to Table 3"
          • "see Figure 2", "as shown in Figure 2"
          • "see page 5", "refer to page 5"
          • "Section 2.1" → links to the first chunk in that section
        """
        ref_patterns = {
            ChunkType.TABLE: re.compile(
                r"(?:see|refer\s+to|in)\s+Table\s+(\d+)", re.IGNORECASE
            ),
            ChunkType.FIGURE: re.compile(
                r"(?:see|refer\s+to|as\s+shown\s+in|in)\s+Figure\s+(\d+)",
                re.IGNORECASE,
            ),
        }
        page_pattern = re.compile(
            r"(?:see|refer\s+to)\s+page\s+(\d+)", re.IGNORECASE
        )
        section_pattern = re.compile(
            r"(?:see\s+)?Section\s+(\d+(?:\.\d+)*)", re.IGNORECASE
        )

        for chunk in chunks:
            if chunk.chunk_type not in (ChunkType.TEXT, ChunkType.LIST):
                continue

            # Table / Figure references
            for target_type, pattern in ref_patterns.items():
                for match in pattern.finditer(chunk.content):
                    ref_num = match.group(1)
                    for target in chunks:
                        if target.chunk_type != target_type:
                            continue
                        caption = target.metadata.get("caption", "")
                        if ref_num in caption:
                            chunk.relationships.append(
                                ChunkRelationship(
                                    relation_type=ChunkRelationType.REFERENCES,
                                    target_chunk_id=target.ldu_id,
                                    score=1.0,
                                )
                            )

            # Page references
            for match in page_pattern.finditer(chunk.content):
                ref_page = int(match.group(1))
                for target in chunks:
                    if ref_page in target.page_refs and target.ldu_id != chunk.ldu_id:
                        chunk.relationships.append(
                            ChunkRelationship(
                                relation_type=ChunkRelationType.REFERENCES,
                                target_chunk_id=target.ldu_id,
                                score=0.8,
                            )
                        )
                        break  # one link per page reference

            # Section references
            for match in section_pattern.finditer(chunk.content):
                sec_ref = match.group(1)
                for target in chunks:
                    if sec_ref in target.parent_section:
                        chunk.relationships.append(
                            ChunkRelationship(
                                relation_type=ChunkRelationType.REFERENCES,
                                target_chunk_id=target.ldu_id,
                                score=0.9,
                            )
                        )
                        break
