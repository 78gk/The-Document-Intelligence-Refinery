import os
import argparse
import yaml
from pathlib import Path
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import SemanticChunker
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryInterfaceAgent

def main():
    parser = argparse.ArgumentParser(description="The Document Intelligence Refinery")
    parser.add_argument("--doc", type=str, required=True, help="Path to the document to process")
    parser.add_argument("--rules", type=str, default="rubric/extraction/rules.yaml", help="Path to rules.yaml")
    parser.add_argument("--query", type=str, help="Optional query to ask about the document")
    args = parser.parse_args()

    # Initialize Agents
    triage_agent = TriageAgent(args.rules)
    with open(args.rules, 'r') as f:
        config = yaml.safe_load(f)
    
    extractor = ExtractionRouter(config)
    chunker = SemanticChunker(config)
    indexer = PageIndexBuilder(config)
    query_agent = QueryInterfaceAgent(config)

    print(f"--- Stage 1: Triage ---")
    profile = triage_agent.triage(args.doc)
    print(f"Document Profile: {profile.origin_type}, {profile.layout_complexity}")

    print(f"\n--- Stage 2: Extraction ---")
    extracted_doc = extractor.extract(args.doc, profile)
    print(f"Extraction Strategy: {extracted_doc.extraction_strategy}, Confidence: {extracted_doc.confidence_score}")

    print(f"\n--- Stage 3: Chunking ---")
    ldus = chunker.chunk(extracted_doc)
    print(f"Generated {len(ldus)} Logical Document Units (LDUs)")

    print(f"\n--- Stage 4: Indexing ---")
    page_index = indexer.build(extracted_doc)
    print(f"PageIndex Tree built with {len(page_index.root.child_sections)} main sections")

    print(f"\n--- Stage 5: Ingestion & Query ---")
    query_agent.ingest_ldus(profile.doc_id, ldus)
    query_agent.load_page_index(profile.doc_id)
    
    if args.query:
        print(f"\nQuery: {args.query}")
        result = query_agent.answer_query(profile.doc_id, args.query)
        print(f"Answer: {result['answer']}")
        print(f"Provenance: {result['provenance']}")

if __name__ == "__main__":
    main()
