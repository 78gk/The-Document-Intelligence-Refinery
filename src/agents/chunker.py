import hashlib
from typing import List, Dict, Any
from ..models.document import ExtractedDocument, TableBlock, TextBlock, FigureBlock, BlockType
from ..models.refinery_models import (
    LDU,
    ChunkType,
    ChunkRelationship,
    ChunkRelationType,
)

class SemanticChunker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        chunk_cfg = config['chunking']
        self.max_tokens = int(chunk_cfg['max_tokens_per_chunk'])
        self.overlap = int(chunk_cfg.get('overlap_tokens', 0))
        self.rules = chunk_cfg.get('rules', {})
        self.list_split_max_tokens = int(self.rules.get('list_split_max_tokens', self.max_tokens))
        self.add_continuation_markers = bool(self.rules.get('add_continuation_markers', True))
        self.propagate_section_headers = bool(self.rules.get('propagate_section_headers', True))
        self._id_counter = 0

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        chunks = []
        previous_chunk_id = None
        
        # Rule 1: Tables are intact LDUs
        for table in doc.tables:
            ldu = self._create_table_ldu(doc, table)
            if previous_chunk_id:
                ldu.relationships.append(
                    ChunkRelationship(
                        relation_type=ChunkRelationType.SIBLING,
                        target_chunk_id=previous_chunk_id,
                    )
                )
            chunks.append(ldu)
            previous_chunk_id = ldu.ldu_id
            
        # Rule 2: Text blocks chunking
        for block in doc.text_blocks:
            text_chunks = self._chunk_text_block(doc, block)
            for text_chunk in text_chunks:
                if previous_chunk_id:
                    text_chunk.relationships.append(
                        ChunkRelationship(
                            relation_type=ChunkRelationType.SIBLING,
                            target_chunk_id=previous_chunk_id,
                        )
                    )
                chunks.append(text_chunk)
                previous_chunk_id = text_chunk.ldu_id

        # Rule 3: Figures are standalone LDUs with captions in metadata
        for figure in doc.figures:
            fig_chunk = self._create_figure_ldu(doc, figure)
            if previous_chunk_id:
                fig_chunk.relationships.append(
                    ChunkRelationship(
                        relation_type=ChunkRelationType.SIBLING,
                        target_chunk_id=previous_chunk_id,
                    )
                )
            chunks.append(fig_chunk)
            previous_chunk_id = fig_chunk.ldu_id
            
        return chunks

    def _next_ldu_id(self, doc_id: str) -> str:
        self._id_counter += 1
        return f"{doc_id}-ldu-{self._id_counter}"

    def _create_table_ldu(self, doc: ExtractedDocument, table: TableBlock) -> LDU:
        # Serialize table for content
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
            token_count=len(content.split()), # Simplification
            content_hash=content_hash,
            parent_section=(table.caption or "table_section"),
            metadata={"caption": table.caption, "confidence": table.confidence}
        )

    def _chunk_text_block(self, doc: ExtractedDocument, block: TextBlock) -> List[LDU]:
        # Simplistic token-based chunking for text blocks (but respecting block boundaries)
        # In a real scenario, this would use a proper tokenizer
        words = block.text.split()
        ldus = []
        token_budget = self._token_budget_for_block(block)

        step = max(1, token_budget - self.overlap) if token_budget > 0 else 1
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
                parent_section=self._parent_section_for_block(block),
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

    def _create_figure_ldu(self, doc: ExtractedDocument, figure: FigureBlock) -> LDU:
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
            parent_section="figures",
            metadata={"caption": caption, "image_path": figure.image_path, "confidence": figure.confidence},
        )

    def _token_budget_for_block(self, block: TextBlock) -> int:
        if block.block_type == BlockType.LIST_ITEM:
            return max(1, self.list_split_max_tokens)
        return max(1, self.max_tokens)

    def _parent_section_for_block(self, block: TextBlock) -> str:
        if self.propagate_section_headers and block.block_type == BlockType.HEADING:
            return "section_heading"
        return "body"
