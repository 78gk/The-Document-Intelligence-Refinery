import argparse
import json
import time
from pathlib import Path

import yaml

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder


def _load_config(rules_path: str) -> dict:
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_document(doc_path: str, rules_path: str, pages: list[int] | None) -> dict:
    cfg = _load_config(rules_path)
    triage = TriageAgent(rules_path)
    router = ExtractionRouter(cfg)
    indexer = PageIndexBuilder(cfg)

    start = time.time()
    profile = triage.triage(doc_path)
    extracted = router.extract(doc_path, profile, pages=pages)
    _ = indexer.build(extracted)

    return {
        "doc_id": profile.doc_id,
        "origin_type": profile.origin_type.value,
        "layout_complexity": profile.layout_complexity.value,
        "strategy": extracted.extraction_strategy.value,
        "confidence": extracted.confidence_score,
        "processing_time": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interim .refinery artifacts for a list of documents")
    parser.add_argument("--rules", default="rubric/extraction/rules.yaml")
    parser.add_argument("--docs", nargs="+", required=True, help="List of document paths")
    parser.add_argument(
        "--pages",
        default="0",
        help="Comma-separated zero-based page indices to process (default: 0). Use 'all' for full document.",
    )
    parser.add_argument("--report", default=".refinery/interim_artifact_run_report.json")
    args = parser.parse_args()

    Path(".refinery").mkdir(parents=True, exist_ok=True)

    pages = None
    if str(args.pages).strip().lower() != "all":
        pages = [int(p.strip()) for p in str(args.pages).split(",") if p.strip()]

    results = []
    for doc in args.docs:
        try:
            results.append(process_document(doc, args.rules, pages))
        except Exception as exc:
            results.append({"doc_path": doc, "error": str(exc)})

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.time(), "results": results}, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
