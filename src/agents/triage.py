import os
import re
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np
import pdfplumber
import yaml

from ..models.profile import CostTier, DocumentProfile, DomainHint, LayoutComplexity, OriginType


class DomainClassifier(Protocol):
    def classify(self, text: str) -> Tuple[DomainHint, str, float]:
        ...


class KeywordDomainClassifier:
    def __init__(
        self,
        domain_keywords: Dict[str, List[str]],
        domain_hint_map: Dict[str, str],
        allowed_domains: List[str],
        min_keyword_hits: int = 1,
        default_domain: str = "general",
        no_match_confidence: float = 0.6,
        hit_confidence_increment: float = 0.1,
        max_confidence: float = 0.99,
    ):
        self.domain_keywords = domain_keywords
        self.domain_hint_map = {k.lower(): v.lower() for k, v in (domain_hint_map or {}).items()}
        self.allowed_domains = {d.lower() for d in (allowed_domains or [])}
        self.min_keyword_hits = min_keyword_hits
        self.default_domain = default_domain.lower()
        self.no_match_confidence = float(no_match_confidence)
        self.hit_confidence_increment = float(hit_confidence_increment)
        self.max_confidence = float(max_confidence)

    def _resolve_hint(self, domain_label: str) -> DomainHint:
        mapped_value = self.domain_hint_map.get(domain_label.lower(), domain_label.lower())
        try:
            return DomainHint(mapped_value)
        except ValueError:
            return DomainHint.OTHER

    def classify(self, text: str) -> Tuple[DomainHint, str, float]:
        normalized_text = text.lower()
        scores: Dict[str, int] = {}

        for domain, keywords in self.domain_keywords.items():
            hits = sum(1 for kw in keywords if kw.lower() in normalized_text)
            if hits >= self.min_keyword_hits:
                scores[domain.lower()] = hits

        if not scores:
            default_hint = self._resolve_hint(self.default_domain)
            return default_hint, self.default_domain, self.no_match_confidence

        best_domain = max(scores, key=scores.get)
        if self.allowed_domains and best_domain not in self.allowed_domains:
            return DomainHint.OTHER, best_domain, self.no_match_confidence

        hint = self._resolve_hint(best_domain)
        confidence = min(
            self.max_confidence,
            self.no_match_confidence + (scores[best_domain] * self.hit_confidence_increment),
        )
        return hint, best_domain, confidence


class TriageAgent:
    def __init__(self, rules_path: str):
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f)

        classification_cfg = self.rules.get("classification", {})
        domain_cfg = classification_cfg.get("domain_classifier", {})
        domain_keywords = classification_cfg.get("domain_classification", {})
        domain_hint_map = classification_cfg.get("domain_hint_map", {})
        domain_taxonomy = classification_cfg.get("domain_taxonomy", list(domain_keywords.keys()))

        self.domain_classifier = KeywordDomainClassifier(
            domain_keywords=domain_keywords,
            domain_hint_map=domain_hint_map,
            allowed_domains=domain_taxonomy,
            min_keyword_hits=int(domain_cfg.get("min_keyword_hits", 1)),
            default_domain=str(domain_cfg.get("default_domain", "general")),
            no_match_confidence=float(domain_cfg.get("no_match_confidence", 0.6)),
            hit_confidence_increment=float(domain_cfg.get("hit_confidence_increment", 0.1)),
            max_confidence=float(domain_cfg.get("max_confidence", 0.99)),
        )

    def triage(self, doc_path: str) -> DocumentProfile:
        doc_id = Path(doc_path).stem
        with pdfplumber.open(doc_path) as pdf:
            profile_data = self._analyze_pdf(pdf)
            page_count = len(pdf.pages)

        profile = DocumentProfile(
            doc_id=doc_id,
            origin_type=profile_data["origin_type"],
            layout_complexity=profile_data["layout_complexity"],
            language=profile_data.get("language", "unknown"),
            language_confidence=profile_data.get("language_confidence", 0.0),
            domain_hint=profile_data["domain_hint"],
            domain_label=profile_data.get("domain_label", profile_data["domain_hint"].value),
            extraction_cost_estimate=profile_data["cost_estimate"],
            triage_confidence=profile_data.get("triage_confidence", 1.0),
            metadata={
                "page_count": page_count,
                "file_size_kb": os.path.getsize(doc_path) / 1024,
                "analysis": profile_data["analysis_details"],
            },
        )

        output_dir = Path(".refinery/profiles")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            f.write(profile.model_dump_json(indent=2))

        return profile

    def _analyze_pdf(self, pdf: pdfplumber.PDF) -> Dict[str, Any]:
        pages = pdf.pages
        if not pages:
            return self._empty_analysis()

        classification_cfg = self.rules["classification"]
        origin_cfg = classification_cfg["origin_types"]
        layout_cfg = classification_cfg["layout_complexity"]
        confidence_cfg = classification_cfg.get("triage_confidence_weights", {})

        total_chars = 0.0
        total_images = 0.0
        form_field_pages = 0
        multi_column_signals = 0
        table_density_signals: List[float] = []
        image_area_signals: List[float] = []
        page_char_density: List[float] = []
        whitespace_ratios: List[float] = []
        font_coverage_by_page: List[float] = []
        scanned_like_pages = 0
        digital_like_pages = 0

        max_sample_pages = int(classification_cfg["max_sample_pages"])
        sample_pages = pages[:max_sample_pages]
        num_sample_pages = len(sample_pages)

        multi_column_min_chars = int(layout_cfg.get("multi_column_min_chars", 100))
        min_char_width_default = float(layout_cfg.get("min_char_width_default", 1.0))
        scanned_requires_image = bool(origin_cfg.get("scanned_page_requires_image_signal", False))

        for page in sample_pages:
            chars = page.chars
            images = page.images
            total_chars += len(chars)
            total_images += len(images)
            page_area = float(page.width) * float(page.height)
            if page_area <= 0:
                continue

            annots = getattr(page, "annots", None) or []
            if annots:
                form_field_pages += 1

            char_density = len(chars) / page_area
            page_char_density.append(char_density)

            used_text_area = sum(
                max(
                    0.0,
                    (float(c.get("x1", 0.0)) - float(c.get("x0", 0.0)))
                    * (float(c.get("bottom", 0.0)) - float(c.get("top", 0.0))),
                )
                for c in chars
            )
            whitespace_ratios.append(float(np.clip(1.0 - (used_text_area / page_area), 0.0, 1.0)))

            font_coverage = (len([c for c in chars if c.get("fontname")]) / len(chars)) if chars else 0.0
            font_coverage_by_page.append(font_coverage)

            if len(chars) >= multi_column_min_chars:
                x_coords = sorted([c["x0"] for c in chars])
                gaps = np.diff(x_coords)
                median_width = np.median([c.get("width", min_char_width_default) for c in chars]) if chars else min_char_width_default
                if gaps.size > 0 and np.max(gaps) > median_width * float(layout_cfg["multi_column_threshold"]):
                    multi_column_signals += 1

            tables = page.find_tables()
            if tables:
                table_area = sum([(t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]) for t in tables])
                table_density_signals.append(table_area / page_area)

            if images:
                image_area = sum([float(img.get("width", 0.0)) * float(img.get("height", 0.0)) for img in images])
                image_area_signals.append(image_area / page_area)

            page_image_density = image_area_signals[-1] if image_area_signals else 0.0
            low_density_signal = char_density <= float(origin_cfg["scanned_char_density_max"])
            low_font_with_image_signal = (
                font_coverage < float(origin_cfg["font_coverage_min"])
                and page_image_density > float(origin_cfg["scanned_image_density_min"])
            )
            scanned_page_signal = (low_density_signal and page_image_density > 0.0) if scanned_requires_image else (low_density_signal or low_font_with_image_signal)

            if scanned_page_signal:
                scanned_like_pages += 1
            else:
                digital_like_pages += 1

        avg_chars_per_page = total_chars / max(1, num_sample_pages)
        avg_images_per_page = total_images / max(1, num_sample_pages)
        avg_table_density = float(np.mean(table_density_signals)) if table_density_signals else 0.0
        avg_image_density = float(np.mean(image_area_signals)) if image_area_signals else 0.0
        avg_char_density = float(np.mean(page_char_density)) if page_char_density else 0.0
        avg_whitespace_ratio = float(np.mean(whitespace_ratios)) if whitespace_ratios else 1.0
        font_coverage = float(np.mean(font_coverage_by_page)) if font_coverage_by_page else 0.0

        scant_threshold = float(origin_cfg["scanned_if_chars_less_than"])
        mixed_page_ratio_min = float(origin_cfg["mixed_page_ratio_min"])
        origin_type = OriginType.NATIVE_DIGITAL

        if form_field_pages / max(1, num_sample_pages) >= float(origin_cfg["form_fillable_annotation_ratio"]):
            origin_type = OriginType.FORM_FILLABLE
        elif (
            scanned_like_pages / max(1, num_sample_pages) >= mixed_page_ratio_min
            and digital_like_pages / max(1, num_sample_pages) >= mixed_page_ratio_min
        ):
            origin_type = OriginType.MIXED
        elif avg_chars_per_page < scant_threshold and avg_image_density >= float(origin_cfg["scanned_image_density_min"]):
            origin_type = OriginType.SCANNED_IMAGE
        elif avg_char_density <= float(origin_cfg["scanned_char_density_max"]):
            origin_type = OriginType.SCANNED_IMAGE
        elif font_coverage < float(origin_cfg["font_coverage_min"]) and avg_whitespace_ratio > float(origin_cfg["scanned_whitespace_ratio_min"]):
            origin_type = OriginType.SCANNED_IMAGE

        layout_complexity = LayoutComplexity.SINGLE_COLUMN
        multi_column_ratio = multi_column_signals / max(1, num_sample_pages)
        if multi_column_ratio > float(layout_cfg["multi_column_signal_ratio"]):
            layout_complexity = LayoutComplexity.MULTI_COLUMN
        elif avg_table_density > float(layout_cfg["table_heavy_density"]):
            layout_complexity = LayoutComplexity.TABLE_HEAVY
        elif avg_image_density > float(layout_cfg["figure_heavy_density"]):
            layout_complexity = LayoutComplexity.FIGURE_HEAVY
        elif avg_table_density > float(layout_cfg["mixed_table_density_min"]) and avg_image_density > float(layout_cfg["mixed_figure_density_min"]):
            layout_complexity = LayoutComplexity.MIXED

        text_sample = "".join([p.extract_text() or "" for p in sample_pages])
        domain_hint, domain_label, domain_confidence = self.domain_classifier.classify(text_sample)
        language, language_confidence = self._detect_language(text_sample, classification_cfg.get("language_detection", {}))

        cost_estimate = CostTier.FAST_TEXT_SUFFICIENT
        if origin_type in (OriginType.SCANNED_IMAGE, OriginType.FORM_FILLABLE):
            cost_estimate = CostTier.NEEDS_VISION_MODEL
        elif origin_type == OriginType.MIXED or layout_complexity != LayoutComplexity.SINGLE_COLUMN:
            cost_estimate = CostTier.NEEDS_LAYOUT_MODEL

        density_target = float(origin_cfg.get("native_char_density_target", 0.0002))
        base_confidence = (
            float(confidence_cfg.get("char_density", 0.3))
            * float(np.clip(avg_char_density / max(density_target, 1e-6), 0.0, 1.0))
            + float(confidence_cfg.get("font_coverage", 0.3)) * font_coverage
            + float(confidence_cfg.get("image_signal", 0.2)) * float(np.clip(1.0 - avg_image_density, 0.0, 1.0))
            + float(confidence_cfg.get("domain_classifier", 0.2)) * domain_confidence
        )

        variance_cfg = classification_cfg.get("triage_confidence_variance", {})
        variance_weight = float(variance_cfg.get("weight", 0.15))
        eps = float(variance_cfg.get("epsilon", 1e-6))
        max_penalty = float(variance_cfg.get("max_penalty", 0.5))
        page_variance = {
            "char_density_cv": self._coefficient_of_variation(page_char_density, eps),
            "font_coverage_cv": self._coefficient_of_variation(font_coverage_by_page, eps),
            "image_density_cv": self._coefficient_of_variation(image_area_signals, eps),
        }
        raw_variance_penalty = float(np.mean(list(page_variance.values()))) if page_variance else 0.0
        variance_penalty = float(np.clip(raw_variance_penalty, 0.0, max_penalty))
        triage_confidence = float(
            np.clip(
                base_confidence * (1.0 - (variance_weight * variance_penalty)),
                0.0,
                1.0,
            )
        )

        return {
            "origin_type": origin_type,
            "layout_complexity": layout_complexity,
            "domain_hint": domain_hint,
            "domain_label": domain_label,
            "language": language,
            "language_confidence": language_confidence,
            "cost_estimate": cost_estimate,
            "triage_confidence": triage_confidence,
            "analysis_details": {
                "avg_chars_per_page": avg_chars_per_page,
                "avg_char_density": avg_char_density,
                "avg_whitespace_ratio": avg_whitespace_ratio,
                "avg_images_per_page": avg_images_per_page,
                "multi_column_signals": multi_column_signals,
                "avg_table_density": avg_table_density,
                "avg_image_density": avg_image_density,
                "font_coverage": font_coverage,
                "form_field_pages": form_field_pages,
                "scanned_like_pages": scanned_like_pages,
                "digital_like_pages": digital_like_pages,
                "variance_penalty": variance_penalty,
                "page_variance": page_variance,
                "language_confidence": language_confidence,
            },
        }

    def _empty_analysis(self) -> Dict[str, Any]:
        return {
            "origin_type": OriginType.SCANNED_IMAGE,
            "layout_complexity": LayoutComplexity.SINGLE_COLUMN,
            "domain_hint": DomainHint.GENERAL,
            "domain_label": DomainHint.GENERAL.value,
            "cost_estimate": CostTier.NEEDS_VISION_MODEL,
            "triage_confidence": 0.0,
            "analysis_details": {"error": "empty_document"},
        }

    def _detect_language(self, text: str, cfg: Dict[str, Any]) -> Tuple[str, float]:
        default_language = str(cfg.get("default_language", "unknown")).lower()
        default_confidence = float(cfg.get("default_confidence", 0.2))
        max_confidence = float(cfg.get("max_confidence", 0.99))
        if not text or not text.strip():
            return default_language, default_confidence

        lowered = text.lower()
        keyword_map = cfg.get(
            "keyword_map",
            {
                "en": ["the", "and", "of", "with", "for"],
                "es": ["el", "la", "de", "que", "para"],
                "fr": ["le", "la", "de", "et", "pour"],
            },
        )
        total_hits = 0
        language_hits: Dict[str, int] = {}

        for lang, keywords in keyword_map.items():
            hits = 0
            for keyword in keywords:
                pattern = rf"\b{re.escape(keyword.lower())}\b"
                hits += len(re.findall(pattern, lowered))
            if hits > 0:
                language_hits[lang.lower()] = hits
                total_hits += hits

        if not language_hits:
            return default_language, default_confidence

        best_language = max(language_hits, key=language_hits.get)
        share = language_hits[best_language] / max(1, total_hits)
        confidence = float(np.clip(default_confidence + (0.8 * share), 0.0, max_confidence))
        return best_language, confidence

    def _coefficient_of_variation(self, values: List[float], epsilon: float) -> float:
        if not values:
            return 0.0
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        if mean <= epsilon:
            return 0.0
        return float(np.std(arr) / (mean + epsilon))
