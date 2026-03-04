import json
from pathlib import Path
from typing import List, Dict, Any
from ..models.document import ExtractedDocument, BlockType
from ..models.refinery_models import PageIndex, SectionNode, ChunkType

class PageIndexBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build(self, doc: ExtractedDocument) -> PageIndex:
        # Simplistic section detection based on heading detection in blocks
        # In prod, this would use the Layout model's hierarchy
        
        root_sections = []
        current_section = None
        
        # Group text blocks into potential sections
        # This is a placeholder for real hierarchical detection
        for block in doc.text_blocks:
            if block.block_type == BlockType.HEADING or len(block.text) < 100: # Heuristic for headings
                if current_section:
                    root_sections.append(current_section)
                
                current_section = SectionNode(
                    node_id=f"{doc.doc_id}-sec-{len(root_sections)+1}",
                    title=block.text[:50],
                    page_start=block.bbox.page,
                    page_end=block.bbox.page,
                    level=1,
                    summary="[SIMULATED LLM SUMMARY]",
                    data_types_present=[ChunkType.TEXT]
                )
            elif current_section:
                current_section.page_end = block.bbox.page
        
        if current_section:
            root_sections.append(current_section)

        # Ensure we have at least one root node
        if not root_sections:
            root_sections.append(SectionNode(
                node_id=f"{doc.doc_id}-sec-1",
                title="Full Document",
                page_start=1,
                page_end=doc.text_blocks[-1].bbox.page if doc.text_blocks else 1,
                level=1,
                summary="Complete document overview",
                data_types_present=[ChunkType.TEXT]
            ))

        root_node = SectionNode(
            node_id=f"{doc.doc_id}-root",
            title=doc.doc_id,
            page_start=1,
            page_end=root_sections[-1].page_end if root_sections else 1,
            level=0,
            child_sections=root_sections,
            summary="Document root navigation tree",
            data_types_present=[ChunkType.TEXT, ChunkType.TABLE, ChunkType.FIGURE]
        )

        page_index = PageIndex(doc_id=doc.doc_id, root=root_node)
        
        # Save output
        output_dir = Path(".refinery/pageindex")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{doc.doc_id}.json", 'w') as f:
            f.write(page_index.model_dump_json(indent=2))
            
        return page_index
    
    def _generate_summaries(self, sections: List[SectionNode]):
        # This would call an LLM to generate 2-3 sentence summaries
        pass
