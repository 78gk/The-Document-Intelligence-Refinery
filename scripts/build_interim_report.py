from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def money(x: float) -> str:
    return f"${x:,.2f}"


def sec(x: float) -> str:
    return f"{x:,.2f}s"


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_rules(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_profiles(profiles_dir: Path) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for p in profiles_dir.glob("*.json"):
        try:
            profiles[p.stem] = read_json(p)
        except Exception:
            # best-effort for report
            continue
    return profiles


def index_ledger_by_doc(ledger: List[dict], profiles: Dict[str, dict]) -> Dict[str, dict]:
    by_doc: Dict[str, dict] = {}
    for entry in ledger:
        doc_id = entry.get("doc_id")
        if not doc_id:
            continue
        if doc_id in profiles:
            by_doc[doc_id] = entry
    return by_doc


def build_cost_rows(docs: List[str], ledger_by_doc: Dict[str, dict], profiles: Dict[str, dict], rules: dict) -> List[dict]:
    cost_a = safe_float(rules["extraction"]["fast_text"].get("estimated_cost_per_page", 0.0))
    cost_b = safe_float(rules["extraction"]["layout_aware"].get("estimated_cost_per_page", 0.0))
    cost_c = safe_float(rules["extraction"]["vision"].get("estimated_cost_per_page", 0.0))
    cap_c = safe_float(rules["extraction"]["vision"].get("budget_cap_per_doc", 0.0))

    # The interim ledger run processed first page only for each document.
    # We report observed 1-page processing time and a linear full-doc projection.
    rows: List[dict] = []
    for doc_id in docs:
        prof = profiles[doc_id]
        led = ledger_by_doc[doc_id]
        page_count = int(prof.get("metadata", {}).get("page_count") or 1)
        pages_at_vision = page_count
        if cost_c > 0 and cap_c > 0:
            pages_at_vision = min(page_count, int(cap_c / cost_c))

        est_cost_a = cost_a * page_count
        est_cost_b = cost_b * page_count
        est_cost_c = cost_c * max(1, pages_at_vision) if cost_c > 0 else 0.0

        run_pages = int(led.get("pages_processed") or 1)
        run_time = safe_float(led.get("processing_time"), 0.0)
        time_per_page = run_time / max(1, run_pages)
        est_time_linear = time_per_page * page_count

        analysis = prof.get("metadata", {}).get("analysis", {})
        rows.append(
            {
                "doc_id": doc_id,
                "origin": prof.get("origin_type"),
                "layout": prof.get("layout_complexity"),
                "domain": prof.get("domain_hint"),
                "pages": page_count,
                "triage_conf": safe_float(prof.get("triage_confidence"), 0.0),
                "char_density": safe_float(analysis.get("avg_char_density"), 0.0),
                "img_density": safe_float(analysis.get("avg_image_density"), 0.0),
                "table_density": safe_float(analysis.get("avg_table_density"), 0.0),
                "strategy_used": led.get("strategy"),
                "confidence": safe_float(led.get("confidence"), 0.0),
                "time_1p": run_time,
                "est_cost_a": est_cost_a,
                "est_cost_b": est_cost_b,
                "est_cost_c": est_cost_c,
                "vision_pages": pages_at_vision,
                "est_time_linear": est_time_linear,
            }
        )
    return rows


def main() -> None:
    rules = load_rules(Path("rubric/extraction/rules.yaml"))
    ledger = read_json(Path(".refinery/extraction_ledger.json"))
    profiles = load_profiles(Path(".refinery/profiles"))
    ledger_by_doc = index_ledger_by_doc(ledger, profiles)

    docs = sorted(ledger_by_doc.keys())

    # For the narrative: explicit examples per class from the corpus.
    # We reference docs from the committed 12-doc ledger subset.
    examples = {
        # A — native financial
        "A": ["Annual_Report_JUNE-2023", "tax_expenditure_ethiopia_2021_22"],
        # B — scanned legal/procedural (note: scanned docs often have zero text stream so domain_hint can degrade to "general")
        "B": [
            "2013-E.C-Procurement-information",
            "Security_Vulnerability_Disclosure_Standard_Procedure_1",
            "Audit Report - 2023",
        ],
        # C — mixed assessment / mixed-mode
        "C": [
            "CBE ANNUAL REPORT 2023-24",
            "fta_performance_survey_final_report_2022",
            "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF",
        ],
        # D — table-heavy fiscal
        "D": ["Consumer Price Index August 2025"],
    }

    def fmt_signal(doc_id: str) -> str:
        prof = profiles.get(doc_id, {})
        analysis = prof.get("metadata", {}).get("analysis", {})
        return (
            f"char_density={safe_float(analysis.get('avg_char_density')):.6f}, "
            f"image_density={safe_float(analysis.get('avg_image_density')):.3f}, "
            f"table_density={safe_float(analysis.get('avg_table_density')):.3f}, "
            f"font_coverage={safe_float(analysis.get('font_coverage')):.2f}"
        )

    def example_metrics_ul(doc_ids: List[str]) -> str:
        items = []
        for d in doc_ids:
            if d not in profiles:
                continue
            p = profiles[d]
            items.append(
                f"<li><code>{html_escape(d)}</code> — {html_escape(str(p.get('origin_type')))} / {html_escape(str(p.get('layout_complexity')))} — <span class='mono'>{html_escape(fmt_signal(d))}</span></li>"
            )
        return "<ul class='small'>" + "".join(items) + "</ul>"
    # Build cost table rows from the committed ledger subset.
    cost_rows = build_cost_rows(docs, ledger_by_doc, profiles, rules)

    # Load inline SVG diagrams.
    arch_svg = Path("scripts/report_assets_architecture.svg").read_text(encoding="utf-8")
    tree_svg = Path("scripts/report_assets_decision_tree.svg").read_text(encoding="utf-8")

    today = dt.datetime.now().strftime("%Y-%m-%d")

    # Config thresholds for narrative.
    fast_thr = safe_float(rules["extraction"]["fast_text"].get("min_confidence_threshold", 0.8))
    layout_thr = safe_float(rules["extraction"]["layout_aware"].get("min_confidence_threshold", 0.7))
    global_thr = safe_float(rules["extraction"].get("graceful_degradation_threshold", 0.8))

    cost_a = safe_float(rules["extraction"]["fast_text"].get("estimated_cost_per_page", 0.0))
    cost_b = safe_float(rules["extraction"]["layout_aware"].get("estimated_cost_per_page", 0.0))
    cost_c = safe_float(rules["extraction"]["vision"].get("estimated_cost_per_page", 0.0))
    cap_c = safe_float(rules["extraction"]["vision"].get("budget_cap_per_doc", 0.0))

    def tr(cells: List[str], header: bool = False) -> str:
        tag = "th" if header else "td"
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

    # Build cost table HTML.
    cost_table = [
        "<table>",
        "<thead>",
        tr(
            [
                "Document",
                "Origin",
                "Layout",
                "Domain",
                "Pages",
                "Obs. strategy",
                "Obs. conf",
                "Time (1p)",
                "Est $ Tier A",
                "Est $ Tier B",
                "Est $ Tier C (cap)",
                "Vision pages",
                "Linear time est.",
            ],
            header=True,
        ),
        "</thead>",
        "<tbody>",
    ]

    for r in sorted(cost_rows, key=lambda x: (x["origin"], x["layout"], x["doc_id"])):
        cost_table.append(
            tr(
                [
                    html_escape(r["doc_id"]),
                    html_escape(str(r["origin"])) ,
                    html_escape(str(r["layout"])) ,
                    html_escape(str(r["domain"])) ,
                    f"<span class='right'>{r['pages']}</span>",
                    html_escape(str(r["strategy_used"])) ,
                    f"{r['confidence']:.3f}",
                    sec(r["time_1p"]),
                    money(r["est_cost_a"]),
                    money(r["est_cost_b"]),
                    money(r["est_cost_c"]),
                    str(r["vision_pages"]),
                    sec(r["est_time_linear"]),
                ]
            )
        )

    cost_table.extend(["</tbody>", "</table>"])

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Interim Report — The Document Intelligence Refinery</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --line: #e2e8f0;
      --soft: #f8fafc;
      --accent: #1f6feb;
      --accent2: #0f766e;
      --warn: #b45309;
    }}
    @page {{ size: A4; margin: 18mm; }}
    html, body {{ color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
    body {{ line-height: 1.5; }}
    h1 {{ font-size: 26px; margin: 0.4rem 0 0.4rem; }}
    h2 {{ font-size: 18px; margin-top: 1.4rem; padding-top: 0.4rem; border-top: 2px solid var(--line); }}
    h3 {{ font-size: 14px; color: var(--accent2); margin-top: 1rem; }}
    .muted {{ color: var(--muted); }}
    .pill {{ display:inline-block; padding:2px 10px; border:1px solid var(--line); border-radius:999px; font-size:12px; color: var(--muted); background: white; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .card {{ border: 1px solid var(--line); border-radius: 12px; padding: 12px 12px; background: white; }}
    .callout {{ border-left: 4px solid var(--accent); background: var(--soft); padding: 10px 12px; border-radius: 10px; }}
    .warn {{ border-left-color: var(--warn); }}
    code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 5px; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 11.5px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; vertical-align: top; }}
    th {{ text-align: left; color: var(--muted); font-weight: 650; }}
    .svg-wrap {{ border:1px solid var(--line); border-radius: 14px; padding: 10px; background: white; }}
    .page-break {{ break-before: page; page-break-before: always; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; }}
    .small {{ font-size: 11px; }}
  </style>
</head>
<body>

<section>
  <div class="pill">B1W3 Interim Report (Single PDF)</div>
  <h1>The Document Intelligence Refinery — Interim Report</h1>
  <div class="muted">Author: <b>Kirubel Tewodros</b> · Date: {today} · Repo: <span class="mono">https://github.com/78gk/The-Document-Intelligence-Refinery</span></div>

  <div class="grid" style="margin-top:12px">
    <div class="card">
      <h3>Executive Summary</h3>
      <ul>
        <li>5-stage pipeline: <b>Triage → Extraction Router → Chunking → PageIndex → Query</b>.</li>
        <li>Three extraction strategies (A/B/C) with a shared interface and normalized <code>ExtractedDocument</code> output.</li>
        <li>Confidence-gated escalation (A→B, B→C) and an audit ledger capturing routing decisions, confidence, and estimated cost.</li>
      </ul>
    </div>
    <div class="card">
      <h3>Corpus Evidence Used</h3>
      <p class="small muted">This report references the committed interim artifacts:</p>
      <ul class="small">
        <li><code>.refinery/profiles/*.json</code> — triage signals + classification</li>
        <li><code>.refinery/pageindex/*.json</code> — navigational structure</li>
        <li><code>.refinery/extraction_ledger.json</code> — strategy used, confidence, time, and cost</li>
      </ul>
    </div>
  </div>
</section>

<section id="domain" class="page-break">
  <h2>1) Domain Notes — Failure Modes & Extraction Strategy Decision Tree</h2>

  <div class="callout">
    <b>What this section proves:</b> (1) empirical failure modes across four classes (A/B/C/D), (2) an explicit decision tree grounded in measurable signals, and (3) a pipeline diagram showing where quality checks occur.
  </div>

  <h3>1.1 Digital vs. Scanned: observable criteria (from our corpus)</h3>
  <div class="grid">
    <div class="card">
      <b>Native digital indicators</b>
      <ul class="small">
        <li>Non-zero <code>avg_char_density</code> and high <code>font_coverage</code> (≈1.0).</li>
        <li>Text stream extractable: pages produce non-empty text without OCR.</li>
        <li>Often lower image dominance (<code>avg_image_density</code> is moderate/low).</li>
      </ul>
      <div class="callout small">
        Example: <code>tax_expenditure_ethiopia_2021_22</code> is <b>native_digital</b> with strong char density + font coverage.
      </div>
    </div>
    <div class="card">
      <b>Scanned indicators</b>
      <ul class="small">
        <li>Near-zero <code>avg_char_density</code> and low/zero <code>font_coverage</code>.</li>
        <li>High <code>avg_image_density</code> (often near 1.0): pages are raster images.</li>
        <li>Text extraction yields empty strings; OCR/VLM required.</li>
      </ul>
      <div class="callout warn small">
        Example: <code>2013-E.C-Procurement-information</code> is <b>scanned_image</b> with <code>avg_char_density≈0</code>, <code>avg_image_density≈1</code>.
      </div>
    </div>
  </div>

  <h3>1.2 Failure modes by document class (A/B/C/D) — empirical + technical causes</h3>

  <div class="card">
    <b>Class A — Native financial</b> <span class="pill">Examples: {', '.join(examples['A'])}</span>
    <ul class="small">
      <li><b>Multi-column reading-order collapse:</b> annual reports often use two columns. Naive extraction can interleave columns, breaking sentence continuity and LDU boundaries. Root cause: insufficient layout modeling.</li>
      <li><b>Table semantics loss:</b> financial statements embed structured tables; text-only extraction removes cell boundaries (headers/rows). Root cause: plain text output lacks table normalization.</li>
    </ul>
    <div class="callout small"><b>Mitigation:</b> Route <i>native_digital + multi_column</i> to Strategy B. Escalate A→B if Strategy A confidence <code>{fast_thr}</code>.</div>
    <div class="callout small" style="margin-top:8px"><b>Measured signals:</b>{example_metrics_ul(examples['A'])}</div>
  </div>

  <div class="card" style="margin-top:10px">
    <b>Class B — Scanned legal / procedural</b> <span class="pill">Examples: {', '.join(examples['B'][:2])}</span>
    <ul class="small">
      <li><b>Zero-text extraction:</b> scanned PDFs carry no character stream; fast extractors return empty content. Root cause: raster-only pages.</li>
      <li><b>Clause/list boundary loss:</b> procedural docs use numbering/indentation; OCR may merge lines or drop punctuation, corrupting downstream chunking.</li>
    </ul>
    <div class="callout warn small"><b>Mitigation:</b> Route scanned to Strategy C (Vision). Budget cap is enforced; non-budget failures degrade gracefully with <code>metadata.degraded=true</code> (no silent pass).</div>
    <div class="callout small" style="margin-top:8px"><b>Measured signals:</b>{example_metrics_ul(examples['B'])}</div>
  </div>

  <div class="card" style="margin-top:10px">
    <b>Class C — Mixed assessment / mixed-mode</b> <span class="pill">Examples: {', '.join(examples['C'][:2])}</span>
    <ul class="small">
      <li><b>Document-level classification hides page variance:</b> mixed docs combine digital text pages and image-heavy pages (scanned inserts / full-page figures). Root cause: per-page origin differs.</li>
      <li><b>Form-fillable semantics:</b> some docs include annotation layers; pure text extraction can miss user-facing form semantics. Root cause: content split across layers.</li>
    </ul>
    <div class="callout small"><b>Mitigation:</b> Start with Strategy B for mixed; escalate to Strategy C when Strategy B confidence <code>{layout_thr}</code> or content coverage is insufficient.</div>
    <div class="callout small" style="margin-top:8px"><b>Measured signals:</b>{example_metrics_ul(examples['C'])}</div>
  </div>

  <div class="card" style="margin-top:10px">
    <b>Class D — Table-heavy fiscal</b> <span class="pill">Examples: {', '.join(examples['D'])}</span>
    <ul class="small">
      <li><b>Dense table boundary ambiguity:</b> CPI/statistical tables may include merged headers and multi-line cells; heuristic table detectors may under/over-segment. Root cause: table structure not explicit in PDF stream.</li>
      <li><b>Provenance ambiguity when flattened:</b> without cell-level bboxes, citations become non-auditable. Root cause: losing structured table objects.</li>
    </ul>
    <div class="callout small"><b>Mitigation:</b> Prefer Strategy B for native table-heavy docs; prefer Strategy C for scanned/form-fillable. Preserve tables using <code>TableCell</code> objects with bboxes.</div>
    <div class="callout small" style="margin-top:8px"><b>Measured signals:</b>{example_metrics_ul(examples['D'])}</div>
  </div>

  <h3>1.3 Extraction strategy decision tree (selection + escalation)</h3>
  <div class="svg-wrap">{tree_svg}</div>
  <p class="small muted">Decision logic is configured in <code>rubric/extraction/rules.yaml</code> (selection maps + thresholds + escalation rules).</p>

  <h3>1.4 Pipeline diagram (quality checks + audit points)</h3>
  <div class="svg-wrap">{arch_svg}</div>
</section>

<section id="arch" class="page-break">
  <h2>2) Architecture Diagram — 5-Stage Pipeline with Strategy Routing</h2>

  <div class="callout">
    <b>Non-linearity:</b> the pipeline branches into strategy tiers and can re-enter extraction via escalation; audit logging is cross-cutting.
  </div>

  <div class="svg-wrap">{arch_svg}</div>

  <div class="card" style="margin-top:12px">
    <h3>Stage responsibilities (inputs → outputs)</h3>
    <table>
      <thead>
        {tr(['Stage','Input','Output','Key responsibilities'], header=True)}
      </thead>
      <tbody>
        {tr(['1) Triage','PDF','DocumentProfile','origin_type, layout_complexity, domain_hint, cost tier, triage_confidence'])}
        {tr(['2) Extraction Router','(PDF, DocumentProfile)','ExtractedDocument','Select A/B/C; compute confidence; escalate; log ledger'])}
        {tr(['3) Semantic Chunker','ExtractedDocument','List[LDU]','Constitution rules; content_hash; chunk relationships'])}
        {tr(['4) PageIndex Builder','ExtractedDocument','PageIndex','Recursive SectionNode hierarchy with page ranges'])}
        {tr(['5) Query Interface','(LDUs, PageIndex)','Answer + ProvenanceChain','Navigation + retrieval with citations'])}
      </tbody>
    </table>
  </div>

  <div class="card" style="margin-top:12px">
    <h3>Provenance layer</h3>
    <ul class="small">
      <li><code>.refinery/extraction_ledger.json</code> captures routing_trace, strategy_used, confidence, time, and estimated_cost_usd.</li>
      <li>LDUs carry spatial provenance and hashes; query answers return a <code>ProvenanceChain</code> for auditability.</li>
    </ul>
  </div>
</section>

<section id="cost" class="page-break">
  <h2>3) Cost Analysis — Per-document estimates by strategy tier</h2>

  <div class="grid">
    <div class="card">
      <h3>Numerical estimates (config-driven)</h3>
      <ul class="small">
        <li>Tier A (Fast Text): <b>{money(cost_a)}</b> / page</li>
        <li>Tier B (Layout-Aware): <b>{money(cost_b)}</b> / page</li>
        <li>Tier C (Vision): <b>{money(cost_c)}</b> / page</li>
        <li>Budget cap (Vision): <b>{money(cap_c)}</b> / document</li>
      </ul>
      <div class="callout small">These values live in <code>rules.yaml</code> so budgeting can be tuned without code changes.</div>
    </div>

    <div class="card">
      <h3>Derivation transparency (explicit chain)</h3>
      <ul class="small">
        <li><b>Tier A/B monetary cost:</b> treated as ~$0 (local compute) for interim; we still track <i>processing time</i> empirically via the ledger.</li>
        <li><b>Tier C monetary cost:</b> we model an <i>effective</i> VLM price via:
          <ul>
            <li>estimated tokens/page = <code>{int(rules['extraction']['vision'].get('avg_tokens_per_page',700))}</code></li>
            <li>assumed effective price (vision+image overhead) ≈ <code>$0.70 / 1K tokens</code></li>
            <li>cost/page ≈ tokens/page ÷ 1000 × price ≈ {int(rules['extraction']['vision'].get('avg_tokens_per_page',700))}/1000×0.70 ≈ <b>$0.49</b> ≈ configured <b>{money(cost_c)}</b></li>
          </ul>
        </li>
        <li>Processing time uses observed ledger time for processed pages, and projects linearly to full document length for planning.</li>
        <li>Cost varies across classes via (a) page count and (b) routing: scanned/table-heavy docs trigger higher tiers.</li>
      </ul>
      <div class="callout warn small"><b>Quality ↔ cost link:</b> Tier B preserves tables/figures/reading order; Tier C recovers text from scanned pages when A/B fail.</div>
    </div>
  </div>

  <h3>3.1 Per-document cost & time table (corpus variation)</h3>
  <div class="card">{''.join(cost_table)}</div>

  <h3>3.2 What higher-cost tiers buy us (tied to document types)</h3>
  <div class="card">
    <ul class="small">
      <li><b>A → B:</b> moves from plain text stream to layout-preserved blocks with bounding boxes, plus structured tables and figures. Critical for Class A (multi-column financial) and Class D (table-heavy fiscal).</li>
      <li><b>B → C:</b> recovers content when the PDF has no text stream (scanned images / low font coverage). Critical for Class B scanned procedural docs and mixed-mode inserts in Class C.</li>
      <li><b>Budget control:</b> Tier C enforces <code>budget_cap_per_doc</code> and now degrades gracefully on non-budget failures (so the pipeline never silently returns empty output).</li>
    </ul>

    <div class="callout small"><b>Quality gate:</b> If final confidence remains below <code>{global_thr}</code>, we flag <code>metadata.review_required=true</code> rather than silently passing low-fidelity extraction.</div>
  </div>
</section>

</body>
</html>
"""

    out_path = Path("interim_report.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
