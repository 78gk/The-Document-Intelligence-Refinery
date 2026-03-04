# Context Log (Sequential / Timestamped)

> Project: The Document Intelligence Refinery
> 
> Purpose: Maintain a running, timestamped log of decisions, changes, and validation results.

## 2026-03-04 23:45:46
- Received instruction to analyze codebase intent/patterns, refer to `TRP1 Challenge Week 3_ The Document Intelligence Refinery.docx.md`, and assess readiness for Wednesday (Day 3) interim submissions.

## 2026-03-04 23:47:00
- Initial test run (`pytest -q`) failed because `pytest` command was not on PATH; switched to `python -m pytest`.
- Found prior logs (`pytest_output.txt`, `test_output.txt`) were being mistakenly collected as tests (collection errors).

## 2026-03-04 23:49:00
- Fixed packaging so imports match tests and challenge scaffold:
  - Added `src/__init__.py` and `__init__.py` files to `src/agents`, `src/models`, `src/strategies`.
  - Added `requires-python = ">=3.10"` to `pyproject.toml`.

## 2026-03-04 23:52:00
- Updated `pytest.ini` to prevent non-test artifacts from being collected:
  - `python_files = test_*.py`
  - Ignored `test_output.txt` and `pytest_output.txt`
  - `norecursedirs = .refinery data`

## 2026-03-04 23:55:00
- Trimmed default dependencies in `pyproject.toml` to keep `pip install -e .` lightweight/fast for interim submission:
  - Kept core: `pydantic`, `pyyaml`, `numpy`, `pdfplumber`
  - Moved dev-only: `pytest` under `[project.optional-dependencies].dev`
  - Removed heavyweight/unused defaults: `chromadb`, `langgraph`
  - Added setuptools package discovery for `src*`

## 2026-03-04 23:58:00
- Validated environment dependency issue: `pdfplumber` was not installed initially due to an interrupted timed install; installed `pdfplumber==0.11.9` and dependencies.

## 2026-03-05 00:01:00
- Validation: `python -m pytest -q` => **18 passed**.
- Current status: codebase test suite is green.

---

## 2026-03-05 00:05:00
- Fixed a production-run robustness issue:
  - `LayoutAwareExtractor` now assigns globally unique `reading_order` across pages.
  - `VisionExtractor` now degrades gracefully when the VLM path is unavailable (e.g., page rendering fails or Ollama isn’t running) by falling back to `pdfplumber` text extraction and embedded figure detection.
- Reran tests: still green.

## 2026-03-05 00:10:00
- Fixed a real-run validation crash in Vision fallback path:
  - `VisionExtractor._bbox_from_page_image` now clamps negative coords to 0.0 to satisfy `BoundingBox` schema.
- After fix, running `main.py` on `CBE ANNUAL REPORT 2023-24.pdf` yields non-zero LDUs even without Ollama.

## 2026-03-05 00:15:00
- Hardened `LayoutAwareExtractor` against negative PDF coordinates (tables/images) by clamping bboxes to non-negative values.
- After fix, `main.py` demo runs on `CBE ANNUAL REPORT 2023-24.pdf` select `layout_aware` with high confidence and produces hundreds of LDUs.
- Tests remain green: `18 passed`.

## Next log entry
- (Add the next significant decision/change here.)
