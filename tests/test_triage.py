import yaml
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.triage import KeywordDomainClassifier, TriageAgent
from src.models.profile import DomainHint, OriginType


def _write_pdf(path: Path, objects: list[str]) -> None:
    out = bytearray()
    out.extend(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj.encode("utf-8"))
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(out)


def _build_blank_pdf(path: Path, pages: int = 1) -> None:
    kids = []
    objects: list[str] = ["", ""]

    for i in range(pages):
        page_obj_num = len(objects) + 1
        content_obj_num = page_obj_num + 1
        kids.append(f"{page_obj_num} 0 R")
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents "
            f"{content_obj_num} 0 R >>"
        )
        objects.append("<< /Length 0 >>\nstream\n\nendstream")

    objects[0] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Count {pages} /Kids [{' '.join(kids)}] >>"
    _write_pdf(path, objects)


def _build_mixed_pdf(path: Path) -> None:
    text_stream = "BT /F1 12 Tf 72 720 Td (Financial report 2024) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Count 2 /Kids [3 0 R 5 0 R] >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(text_stream)} >>\nstream\n{text_stream}\nendstream",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R >>",
        "<< /Length 0 >>\nstream\n\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    _write_pdf(path, objects)


def _build_form_fillable_pdf(path: Path) -> None:
    text_stream = "BT /F1 12 Tf 72 720 Td (Form page) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R /AcroForm 8 0 R >>",
        "<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /Annots [7 0 R] >>",
        f"<< /Length {len(text_stream)} >>\nstream\n{text_stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        "<< /Length 0 >>\nstream\n\nendstream",
        "<< /Type /Annot /Subtype /Widget /FT /Tx /T (Name) /Rect [100 600 300 620] /V () /DA (/Helv 12 Tf 0 g) /P 3 0 R >>",
        "<< /Fields [7 0 R] /DR << /Font << /Helv 5 0 R >> >> /DA (/Helv 12 Tf 0 g) >>",
    ]
    _write_pdf(path, objects)


def _agent_with_relaxed_thresholds(tmp_path: Path) -> TriageAgent:
    base_rules = yaml.safe_load(Path("rubric/extraction/rules.yaml").read_text(encoding="utf-8"))
    base_rules["classification"]["origin_types"]["form_fillable_annotation_ratio"] = 0.01
    base_rules["classification"]["origin_types"]["mixed_page_ratio_min"] = 0.2
    base_rules["classification"]["origin_types"]["scanned_char_density_max"] = 0.00001
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(yaml.safe_dump(base_rules), encoding="utf-8")
    return TriageAgent(str(rules_path))


def test_triage_native_digital():
    rules_path = "rubric/extraction/rules.yaml"
    if not Path(rules_path).exists():
        pytest.skip("rules.yaml not found")

    agent = TriageAgent(rules_path)
    assert agent.rules is not None


def test_origin_type_logic():
    agent = TriageAgent("rubric/extraction/rules.yaml")

    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.chars = [{"text": "a", "fontname": "Arial", "width": 10, "x0": 0}] * 1000
    mock_page.images = []
    mock_page.width = 1000
    mock_page.height = 1000
    mock_page.annots = []
    mock_page.extract_text.return_value = "Financial Report 2024"
    mock_page.find_tables.return_value = []
    mock_pdf.pages = [mock_page]

    analysis = agent._analyze_pdf(mock_pdf)
    assert analysis["origin_type"] == OriginType.NATIVE_DIGITAL
    assert analysis["domain_hint"].value == "financial"
    assert analysis["triage_confidence"] > 0.9


def test_zero_text_document_classifies_scanned(tmp_path):
    pdf_path = tmp_path / "zero_text.pdf"
    _build_blank_pdf(pdf_path, pages=2)

    agent = _agent_with_relaxed_thresholds(tmp_path)
    profile = agent.triage(str(pdf_path))

    assert profile.origin_type == OriginType.SCANNED_IMAGE


def test_mixed_document_classifies_mixed(tmp_path):
    pdf_path = tmp_path / "mixed_doc.pdf"
    _build_mixed_pdf(pdf_path)

    agent = _agent_with_relaxed_thresholds(tmp_path)
    profile = agent.triage(str(pdf_path))

    assert profile.origin_type == OriginType.MIXED


def test_form_fillable_document_detected(tmp_path):
    pdf_path = tmp_path / "form_fillable.pdf"
    _build_form_fillable_pdf(pdf_path)

    agent = _agent_with_relaxed_thresholds(tmp_path)
    profile = agent.triage(str(pdf_path))

    assert profile.origin_type == OriginType.FORM_FILLABLE


def test_domain_classifier_is_config_driven_for_new_label():
    classifier = KeywordDomainClassifier(
        domain_keywords={"energy": ["turbine", "grid"], "general": ["document"]},
        domain_hint_map={"energy": "other", "general": "general"},
        allowed_domains=["energy", "general", "other"],
        min_keyword_hits=1,
        default_domain="general",
    )

    domain_hint, domain_label, _ = classifier.classify("Wind turbine modernization roadmap")
    assert domain_label == "energy"
    assert domain_hint == DomainHint.OTHER


def test_triage_language_detection_not_hardcoded():
    agent = TriageAgent("rubric/extraction/rules.yaml")

    mock_pdf = MagicMock()
    page = MagicMock()
    page.width = 1000
    page.height = 1000
    page.annots = []
    page.images = []
    page.find_tables.return_value = []
    page.chars = [{"text": "x", "fontname": "Arial", "width": 8, "x0": i} for i in range(400)]
    page.extract_text.return_value = "Le rapport de conformite pour accord et politique"
    mock_pdf.pages = [page]

    analysis = agent._analyze_pdf(mock_pdf)
    assert analysis["language"] == "fr"
    assert analysis["language_confidence"] > 0.2


def test_triage_confidence_penalizes_high_variance():
    agent = TriageAgent("rubric/extraction/rules.yaml")

    stable_pdf = MagicMock()
    stable_page = MagicMock()
    stable_page.width = 1000
    stable_page.height = 1000
    stable_page.annots = []
    stable_page.images = []
    stable_page.find_tables.return_value = []
    stable_page.chars = [{"text": "a", "fontname": "Arial", "width": 8, "x0": i} for i in range(500)]
    stable_page.extract_text.return_value = "Technical architecture report and implementation details"
    stable_pdf.pages = [stable_page, stable_page]

    mixed_pdf = MagicMock()
    rich_page = MagicMock()
    rich_page.width = 1000
    rich_page.height = 1000
    rich_page.annots = []
    rich_page.images = []
    rich_page.find_tables.return_value = []
    rich_page.chars = [{"text": "a", "fontname": "Arial", "width": 8, "x0": i} for i in range(500)]
    rich_page.extract_text.return_value = "Technical architecture report and implementation details"

    sparse_page = MagicMock()
    sparse_page.width = 1000
    sparse_page.height = 1000
    sparse_page.annots = []
    sparse_page.images = [{"width": 600.0, "height": 700.0}]
    sparse_page.find_tables.return_value = []
    sparse_page.chars = []
    sparse_page.extract_text.return_value = ""

    mixed_pdf.pages = [rich_page, sparse_page]

    stable = agent._analyze_pdf(stable_pdf)
    mixed = agent._analyze_pdf(mixed_pdf)

    assert stable["triage_confidence"] > mixed["triage_confidence"]
