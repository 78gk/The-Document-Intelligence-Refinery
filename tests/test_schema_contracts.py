import pytest

from src.models.document import BlockType, BoundingBox, ExtractedDocument, TextBlock
from src.models.refinery_models import (
    ChunkRelationType,
    ChunkRelationship,
    ChunkType,
    LDU,
    Provenance,
    ProvenanceChain,
    SectionNode,
)


def test_ldu_requires_core_provenance_fields():
    ldu = LDU(
        ldu_id="doc-ldu-1",
        doc_id="doc",
        content="hello world",
        chunk_type=ChunkType.TEXT,
        page_refs=[1, 1],
        bounding_box=BoundingBox(x0=0, y0=0, x1=10, y1=10, page=1),
        parent_section="intro",
        token_count=2,
        content_hash="1234567890abcdef",
        relationships=[
            ChunkRelationship(
                relation_type=ChunkRelationType.CONTINUATION,
                target_chunk_id="doc-ldu-2",
            )
        ],
    )
    assert ldu.page_refs == [1]
    assert ldu.relationships[0].target_chunk_id == "doc-ldu-2"


def test_section_node_validates_page_ranges():
    with pytest.raises(ValueError):
        SectionNode(
            node_id="n1",
            title="Bad range",
            page_start=3,
            page_end=2,
            level=1,
            summary="x",
        )


def test_provenance_chain_requires_bbox_and_content_hash():
    chain = ProvenanceChain(
        document_name="doc",
        citations=[
            Provenance(
                document_name="doc",
                page_number=1,
                bbox=BoundingBox(x0=1, y0=1, x1=2, y1=2, page=1),
                content_hash="abcdef123456",
            )
        ],
    )
    assert chain.citations[0].bbox.page == 1


def test_extracted_document_derives_top_level_reading_order():
    doc = ExtractedDocument(
        doc_id="doc",
        text_blocks=[
            TextBlock(
                text="A",
                bbox=BoundingBox(x0=0, y0=0, x1=2, y1=2, page=1),
                block_type=BlockType.PARAGRAPH,
                reading_order=1,
                confidence=0.9,
            ),
            TextBlock(
                text="B",
                bbox=BoundingBox(x0=0, y0=2, x1=2, y1=4, page=1),
                block_type=BlockType.PARAGRAPH,
                reading_order=0,
                confidence=0.9,
            ),
        ],
        extraction_strategy="fast_text",
        confidence_score=0.9,
        processing_time=0.01,
    )
    assert doc.reading_order == [0, 1]


def test_extracted_document_rejects_duplicate_reading_order():
    with pytest.raises(ValueError):
        ExtractedDocument(
            doc_id="doc",
            text_blocks=[
                TextBlock(
                    text="A",
                    bbox=BoundingBox(x0=0, y0=0, x1=2, y1=2, page=1),
                    block_type=BlockType.PARAGRAPH,
                    reading_order=0,
                    confidence=0.9,
                ),
                TextBlock(
                    text="B",
                    bbox=BoundingBox(x0=0, y0=2, x1=2, y1=4, page=1),
                    block_type=BlockType.PARAGRAPH,
                    reading_order=0,
                    confidence=0.9,
                ),
            ],
            extraction_strategy="fast_text",
            confidence_score=0.9,
            processing_time=0.01,
        )
