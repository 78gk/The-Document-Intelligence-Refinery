import os
import json
import yaml
import pdfplumber
from pathlib import Path
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import SemanticChunker
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryInterfaceAgent

# Document Selection (3 per class)
CORPUS_DIR = Path("data/data")
TARGET_DOCS = {
    "Class A: Annual Financial Report": [
        "CBE ANNUAL REPORT 2023-24.pdf",
        "CBE Annual Report 2011-12.pdf",
        "CBE Annual Report 2010-11.pdf"
    ],
    "Class B: Scanned Auditor Report": [
        "Audit Report - 2023.pdf",
        "2021_Audited_Financial_Statement_Report.pdf",
        "2022_Audited_Financial_Statement_Report.pdf"
    ],
    "Class C: Technical/Technical Report": [
        "fta_performance_survey_final_report_2022.pdf",
        "2013-E.C-Audit-finding-information.pdf",
        "2013-E.C-Procurement-information.pdf"
    ],
    "Class D: Structured Data Report": [
        "tax_expenditure_ethiopia_2021_22.pdf",
        "Consumer Price Index August 2025.pdf",
        "Consumer Price Index July 2025.pdf"
    ]
}

RULES_PATH = "rubric/extraction/rules.yaml"

def generate():
    with open(RULES_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Agents
    triage_agent = TriageAgent(RULES_PATH)
    extractor = ExtractionRouter(config)
    chunker = SemanticChunker(config)
    indexer = PageIndexBuilder(config)
    query_agent = QueryInterfaceAgent(config)

    qa_examples = []
    processed_count = 0

    for class_name, docs in TARGET_DOCS.items():
        print(f"\n=== Processing {class_name} ===")
        for doc_name in docs:
            doc_path = CORPUS_DIR / doc_name
            if not doc_path.exists():
                print(f"Warning: {doc_name} not found.")
                continue

            print(f"Processing {doc_name}...", flush=True)
            doc_path = CORPUS_DIR / doc_name
            print(f"  Doc path: {doc_path.absolute()}")
            print(f"  Exists: {doc_path.exists()}")
            if not doc_path.exists():
                print(f"  SKIPPING: Not found")
                continue
            try:
                # Run Pipeline
                print("  Triage...", flush=True)
                profile = triage_agent.triage(str(doc_path))
                print(f"  Profile: {profile.origin_type}", flush=True)
                
                # Limit to 20 pages for very large docs to avoid timeouts/memory issues in this env
                # but try to get enough for a good PageIndex
                pages_to_process = None
                with pdfplumber.open(doc_path) as pdf:
                    if len(pdf.pages) > 20:
                        pages_to_process = list(range(20))
                        print(f"  (Large doc: limiting to first 20 pages)")

                extracted_doc = extractor.extract(str(doc_path), profile, pages=pages_to_process)
                ldus = chunker.chunk(extracted_doc)
                page_index = indexer.build(extracted_doc)
                
                query_agent.ingest_ldus(profile.doc_id, ldus)
                query_agent.load_page_index(profile.doc_id)

                # Generate one Q&A per document
                query = "What are the key highlights or findings in this document?"
                if "Financial" in class_name or "Audit" in class_name:
                    query = "What is the total revenue or key financial result mentioned?"
                elif "Tax" in class_name:
                    query = "What is the largest tax expenditure category?"

                answer_data = query_agent.answer_query(profile.doc_id, query)
                qa_examples.append({
                    "document": doc_name,
                    "class": class_name,
                    "query": query,
                    "answer": answer_data["answer"],
                    "provenance": answer_data["provenance"]
                })
                processed_count += 1
            except Exception as e:
                print(f"  ERROR processing {doc_name}: {e}")
                import traceback
                traceback.print_exc()

    # Save Q&A examples
    output_path = Path(".refinery/final_qa_examples.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_examples, f, indent=2)
    
    print(f"\nSuccessfully processed {processed_count} documents.")
    print(f"PageIndex JSONs saved in .refinery/pageindex/")
    print(f"Q&A examples saved in {output_path}")

if __name__ == "__main__":
    generate()
