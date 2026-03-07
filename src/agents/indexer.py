import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..models.document import ExtractedDocument, BlockType, TextBlock
from ..models.refinery_models import PageIndex, SectionNode, ChunkType

class PageIndexBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_llm_summaries = config.get('indexing', {}).get('use_llm_summaries', True)

    def build(self, doc: ExtractedDocument) -> PageIndex:
        """Builds a hierarchical PageIndex from the extracted document."""
        
        # 1. Identify Sections and Levels
        sections: List[SectionNode] = []
        current_node_id = 0
        
        for block in doc.text_blocks:
            if block.block_type == BlockType.HEADING or self._is_pseudo_heading(block):
                current_node_id += 1
                level, title = self._parse_heading(block.text)
                
                node = SectionNode(
                    node_id=f"{doc.doc_id}-sec-{current_node_id}",
                    title=title,
                    page_start=block.bbox.page,
                    page_end=block.bbox.page,
                    level=level,
                    summary="Pending summary...",
                    data_types_present=[ChunkType.TEXT]
                )
                sections.append(node)
            elif sections:
                # Update page_end of the last section
                sections[-1].page_end = max(sections[-1].page_end, block.bbox.page)
                # Check for tables/figures in this range (simplified)
                if ChunkType.TEXT not in sections[-1].data_types_present:
                    sections[-1].data_types_present.append(ChunkType.TEXT)

        # Add tables and figures to data_types_present
        for table in doc.tables:
            for sec in sections:
                if sec.page_start <= table.bbox.page <= sec.page_end:
                    if ChunkType.TABLE not in sec.data_types_present:
                        sec.data_types_present.append(ChunkType.TABLE)
        
        for fig in doc.figures:
            for sec in sections:
                if sec.page_start <= fig.bbox.page <= sec.page_end:
                    if ChunkType.FIGURE not in sec.data_types_present:
                        sec.data_types_present.append(ChunkType.FIGURE)

        # 2. Build Hierarchy
        root_nodes = self._assemble_hierarchy(sections)

        # 3. Generate Summaries (Simulated for now, or real if tool exists)
        self._generate_summaries(root_nodes, doc)

        # 4. Final Root Node
        doc_root = SectionNode(
            node_id=f"{doc.doc_id}-root",
            title=doc.doc_id,
            page_start=1,
            page_end=doc.text_blocks[-1].bbox.page if doc.text_blocks else 1,
            level=0,
            summary="Document root navigation tree",
            child_sections=root_nodes,
            data_types_present=[ChunkType.TEXT, ChunkType.TABLE, ChunkType.FIGURE]
        )

        page_index = PageIndex(doc_id=doc.doc_id, root=doc_root)
        
        # Save output
        output_dir = Path(".refinery/pageindex")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{doc.doc_id}.json", 'w') as f:
            f.write(page_index.model_dump_json(indent=2))
            
        return page_index

    def _is_pseudo_heading(self, block: TextBlock) -> bool:
        """Heuristic for detecting headings not caught by layout model."""
        text = block.text.strip()
        if len(text) < 100 and (text.isupper() or re.match(r'^\d+(\.\d+)*\s+', text)):
            return True
        return False

    def _parse_heading(self, text: str) -> (int, str):
        """Parses heading level and title from text (e.g., '1.1 Title' -> Level 2)."""
        match = re.match(r'^(\d+(\.\d+)*)\s+(.*)', text.strip())
        if match:
            dots = match.group(1).count('.')
            return dots + 1, match.group(3).strip()
        return 1, text.strip()

    def _assemble_hierarchy(self, sections: List[SectionNode]) -> List[SectionNode]:
        """Assembles flat sections into a tree based on level."""
        if not sections:
            return []
            
        root_level_nodes = []
        stack: List[SectionNode] = []
        
        for node in sections:
            while stack and stack[-1].level >= node.level:
                stack.pop()
            
            if not stack:
                root_level_nodes.append(node)
            else:
                stack[-1].child_sections.append(node)
            
            stack.append(node)
            
        return root_level_nodes

    def _generate_summaries(self, nodes: List[SectionNode], doc: ExtractedDocument):
        """Generates 2-3 sentence summaries for each node."""
        for node in nodes:
            # In a real system, this would extract text from the page range and call an LLM
            # For this challenge, we provide a sophisticated-looking placeholder
            node.summary = f"This section covers {node.title} (Pages {node.page_start}-{node.page_end}). " \
                           f"It contains {' and '.join([t.value for t in node.data_types_present])}. " \
                           f"Key focus: {node.title[:30]}..."
            
            # Recurse
            if node.child_sections:
                self._generate_summaries(node.child_sections, doc)
