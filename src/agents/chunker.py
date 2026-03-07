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

class ChunkValidator:
    """Validator to enforce semantic chunking rules."""
    
    @staticmethod
    def validate(chunks: List[LDU]) -> bool:
        for chunk in chunks:
            # Rule 1 & 2: Structural Integrity
            if chunk.chunk_type == ChunkType.TABLE:
                if "caption" not in chunk.metadata:
                    return False
            if chunk.chunk_type == ChunkType.FIGURE:
                if "caption" not in chunk.metadata:
                    return False
            
            # Rule 4: Section Propagation
            if not chunk.parent_section:
                return False
                
        return True

class SemanticChunker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        chunk_cfg = config.get('chunking', {})
        self.max_tokens = int(chunk_cfg.get('max_tokens_per_chunk', 512))
        self.overlap = int(chunk_cfg.get('overlap_tokens', 50))
        self.rules = chunk_cfg.get('rules', {})
        self.list_split_max_tokens = int(self.rules.get('list_split_max_tokens', self.max_tokens))
        self.add_continuation_markers = bool(self.rules.get('add_continuation_markers', True))
        self.propagate_section_headers = bool(self.rules.get('propagate_section_headers', True))
        self._id_counter = 0

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        chunks = []
        current_section = "Document Start"
        previous_chunk_id = None
        
        # We need to process blocks in reading order to maintain section context
        # Combine all blocks and sort by reading order if available
        all_blocks = []
        for b in doc.text_blocks:
            all_blocks.append(('text', b))
        for b in doc.tables:
            all_blocks.append(('table', b))
        for b in doc.figures:
            all_blocks.append(('figure', b))
            
        # Sort by page then reading order/y-coordinate as a fallback
        all_blocks.sort(key=lambda x: (x[1].bbox.page, getattr(x[1], 'reading_order', x[1].bbox.y0)))

        for block_type, block in all_blocks:
            if block_type == 'text':
                if block.block_type == BlockType.HEADING:
                    current_section = block.text.strip()
                
                text_chunks = self._chunk_text_block(doc, block, current_section)
                for text_chunk in text_chunks:
                    self._link_sequential_chunks(text_chunk, previous_chunk_id, chunks)
                    chunks.append(text_chunk)
                    previous_chunk_id = text_chunk.ldu_id
            
            elif block_type == 'table':
                table_chunk = self._create_table_ldu(doc, block, current_section)
                self._link_sequential_chunks(table_chunk, previous_chunk_id, chunks)
                chunks.append(table_chunk)
                previous_chunk_id = table_chunk.ldu_id
                
            elif block_type == 'figure':
                fig_chunk = self._create_figure_ldu(doc, block, current_section)
                self._link_sequential_chunks(fig_chunk, previous_chunk_id, chunks)
                chunks.append(fig_chunk)
                previous_chunk_id = fig_chunk.ldu_id

        # Rule 5: Resolve Cross-references
        self._resolve_cross_references(chunks)
        
        # Validation
        if not ChunkValidator.validate(chunks):
            # In a real system, we'd log a warning or handle this
            pass
            
        return chunks

    def _link_sequential_chunks(self, current_chunk: LDU, previous_chunk_id: Optional[str], all_chunks: List[LDU]):
        if previous_chunk_id:
            current_chunk.relationships.append(
                ChunkRelationship(
                    relation_type=ChunkRelationType.SIBLING,
                    target_chunk_id=previous_chunk_id,
                )
            )

    def _next_ldu_id(self, doc_id: str) -> str:
        self._id_counter += 1
        return f"{doc_id}-ldu-{self._id_counter}"

    def _create_table_ldu(self, doc: ExtractedDocument, table: TableBlock, section: str) -> LDU:
        content = f"Table Caption: {table.caption or 'Untitled Table'}\n"
        for row in table.rows:
            content += " | ".join([cell.text for cell in row]) + "\n"
            
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        return LDU(
            ldu_id=self._next_ldu_id(doc.doc_id),
            doc_id=doc.doc_id,
            content=content,
            chunk_type=ChunkType.TABLE,
            page_refs=[table.bbox.page],
            bounding_box=table.bbox,
            token_count=len(content.split()),
            content_hash=content_hash,
            parent_section=section,
            metadata={"caption": table.caption, "confidence": table.confidence}
        )

    def _chunk_text_block(self, doc: ExtractedDocument, block: TextBlock, section: str) -> List[LDU]:
        words = block.text.split()
        ldus = []
        token_budget = self._token_budget_for_block(block)

        # Ensure lists are kept together if possible
        if block.block_type == BlockType.LIST_ITEM and len(words) < self.list_split_max_tokens:
            token_budget = len(words)

        step = max(1, token_budget - self.overlap) if token_budget > self.overlap else token_budget
        if step <= 0: step = 1
        
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i : i + token_budget])
            if not chunk_text.strip():
                continue
            if self.add_continuation_markers and i > 0:
                chunk_text = f"[CONT] {chunk_text}"
            
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
            ldu = LDU(
                ldu_id=self._next_ldu_id(doc.doc_id),
                doc_id=doc.doc_id,
                content=chunk_text,
                chunk_type=ChunkType.TEXT,
                page_refs=[block.bbox.page],
                bounding_box=block.bbox,
                token_count=len(chunk_text.split()),
                content_hash=content_hash,
                parent_section=section,
                metadata={"block_type": block.block_type.value, "reading_order": block.reading_order}
            )
            
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

    def _create_figure_ldu(self, doc: ExtractedDocument, figure: FigureBlock, section: str) -> LDU:
        caption = figure.caption or "Untitled Figure"
        content = f"Figure Caption: {caption}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return LDU(
            ldu_id=self._next_ldu_id(doc.doc_id),
            doc_id=doc.doc_id,
            content=content,
            chunk_type=ChunkType.FIGURE,
            page_refs=[figure.bbox.page],
            bounding_box=figure.bbox,
            token_count=len(content.split()),
            content_hash=content_hash,
            parent_section=section,
            metadata={"caption": caption, "image_path": figure.image_path, "confidence": figure.confidence},
        )

    def _token_budget_for_block(self, block: TextBlock) -> int:
        if block.block_type == BlockType.LIST_ITEM:
            return max(1, self.list_split_max_tokens)
        return max(1, self.max_tokens)

    def _resolve_cross_references(self, chunks: List[LDU]):
        """Rule 5: Detect 'see Table X' or 'see Figure Y' and link chunks."""
        ref_patterns = {
            ChunkType.TABLE: re.compile(r"see Table (\d+)", re.IGNORECASE),
            ChunkType.FIGURE: re.compile(r"see Figure (\d+)", re.IGNORECASE),
        }
        
        for chunk in chunks:
            if chunk.chunk_type != ChunkType.TEXT:
                continue
                
            for target_type, pattern in ref_patterns.items():
                matches = pattern.findall(chunk.content)
                for match in matches:
                    # Try to find a chunk of target_type whose caption or content matches the reference
                    # This is a heuristic: matching by number in caption
                    for target_chunk in chunks:
                        if target_chunk.chunk_type == target_type:
                            caption = target_chunk.metadata.get("caption", "")
                            if f"Table {match}" in caption or f"Figure {match}" in caption or match in caption:
                                chunk.relationships.append(
                                    ChunkRelationship(
                                        relation_type=ChunkRelationType.REFERENCES,
                                        target_chunk_id=target_chunk.ldu_id,
                                        score=1.0
                                    )
                                )
