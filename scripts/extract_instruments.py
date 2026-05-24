#!/usr/bin/env python3
"""Extract instrument material rows from isometric PDF BOM tables.

The scraper is intentionally strict: rows are emitted only when they are under
OTHER THAN SHOP MATERIALS and then under an INSTRUMENTS category.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber


SECTION_LABEL = "OTHER THAN SHOP MATERIALS"
CATEGORY_LABEL = "INSTRUMENTS"

ROW_COLUMNS = [
    "source_pdf",
    "source_path",
    "page_number",
    "section",
    "category",
    "item_no",
    "qty",
    "tag_no",
    "size",
    "description",
    "material",
    "commodity_code",
    "remarks",
    "raw_text",
    "bbox",
    "confidence",
    "warnings",
]

METADATA_KEYS = [
    "Title",
    "Author",
    "Subject",
    "Creator",
    "Producer",
    "CreationDate",
    "ModDate",
]

CATEGORY_HEADERS = {
    "PIPE",
    "PIPING",
    "FITTINGS",
    "FLANGES",
    "VALVES",
    "GASKETS",
    "BOLTS",
    "SUPPORTS",
    "PIPESUPPORTS",
    "HANGERS",
    "MISC",
    "MISCELLANEOUS",
    "SPECIALTYITEMS",
    "PAINT",
    "INSULATION",
    "SHOPMATERIALS",
    "OTHERTHANSHOPMATERIALS",
}

INSTRUMENT_HEADERS = {"INSTRUMENTS", "INSTRUMENT", "INSTRUMENTATION", "INST"}
HEADER_ALIASES = {
    "ITEM": "item_no",
    "NO": "item_no",
    "QTY": "qty",
    "QUANTITY": "qty",
    "SIZE": "size",
    "NPD": "size",
    "DESCRIPTION": "description",
    "MATERIAL": "material",
    "MATL": "material",
    "CODE": "commodity_code",
    "IDENT": "commodity_code",
    "COMMODITYCODE": "commodity_code",
    "REMARKS": "remarks",
    "REMARK": "remarks",
}

INSTRUMENT_TAG_RE = re.compile(
    r"\b(?:\d+-)?(?:PDT|PIT|PT|PI|LT|LI|LIT|LG|FT|FI|FIT|FE|TT|TI|TE|"
    r"XV|ZS|ZSC|ZSO|LSH|LSL|LS|PSV|PCV|FCV|LCV|TCV|FO|FV|LV|HV|TV|"
    r"PG|TG|LIC|PIC|TIC|FIC|SDV|MOV|AOV|PV|AE|PSE|TDV|VG)-[A-Z0-9]+"
    r"(?:-[A-Z0-9]+)*\b",
    re.IGNORECASE,
)
COMPACT_INSTRUMENT_TAG_RE = re.compile(
    r"(?:\d+-)?(?:PDT|PIT|PT|PI|LT|LI|LIT|LG|FT|FI|FIT|FE|TT|TI|TE|"
    r"XV|ZS|ZSC|ZSO|LSH|LSL|LS|PSV|PCV|FCV|LCV|TCV|FO|FV|LV|HV|TV|"
    r"PG|TG|LIC|PIC|TIC|FIC|SDV|MOV|AOV|PV|AE|PSE|TDV|VG)-\d{3,5}[A-Z]?",
    re.IGNORECASE,
)
FOOTER_OR_TITLE_ANCHORS = {
    "SERVICE",
    "VAPORLIQUID",
    "INSULTYPE",
    "INSULTHK",
    "PAINTCODE",
    "MODULE",
    "PID",
    "UNIT",
    "STRESSCALC",
    "LINENO",
    "DRAWINGNUMBER",
    "DRAWINGNUM",
    "REV",
    "REVDATE",
    "PROJECT",
    "BATTLEGROUND",
    "OXYCHEM",
    "NOTES",
    "WELDS",
}
QTY_RE = re.compile(r"^\d+(?:\.\d+)?$")
ITEM_RE = re.compile(r"^\d{1,4}[A-Z]?$", re.IGNORECASE)
SIZE_RE = re.compile(
    r'^\d+(?:X\d+(?:\.\d+/\d+|/\d+|\.\d+)?)?(?:\.\d+/\d+|/\d+|\.\d+)?"?$',
    re.IGNORECASE,
)


@dataclass
class Line:
    words: list[dict[str, Any]]

    @property
    def text(self) -> str:
        return " ".join(str(word.get("text", "")).strip() for word in self.words).strip()

    @property
    def normalized(self) -> str:
        return normalize(self.text)

    @property
    def top(self) -> float:
        return min(float(word["top"]) for word in self.words)

    @property
    def bottom(self) -> float:
        return max(float(word["bottom"]) for word in self.words)

    @property
    def bbox(self) -> list[float]:
        return [
            min(float(word["x0"]) for word in self.words),
            self.top,
            max(float(word["x1"]) for word in self.words),
            self.bottom,
        ]


@dataclass
class ParsedRow:
    source_pdf: str
    source_path: str
    page_number: int
    item_no: str = ""
    qty: str = ""
    tag_no: str = ""
    size: str = ""
    description: str = ""
    material: str = ""
    commodity_code: str = ""
    remarks: str = ""
    raw_text: str = ""
    bbox: list[float] = field(default_factory=list)
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    wrapped: bool = False
    category_boundary_unclear: bool = False
    column_positions_uncertain: bool = False
    ocr_used: bool = False

    def finalize(self) -> dict[str, Any]:
        warnings = list(dict.fromkeys(self.warnings))
        confidence = 1.0
        if not self.qty:
            confidence -= 0.20
            warnings.append("Missing quantity")
        if not self.description:
            confidence -= 0.25
            warnings.append("Missing description")
        if not self.tag_no:
            confidence -= 0.10
            warnings.append("Missing tag number")
        if self.category_boundary_unclear:
            confidence -= 0.20
            warnings.append("Category boundary unclear")
        if self.column_positions_uncertain:
            confidence -= 0.15
            warnings.append("Column positions uncertain")
        if self.ocr_used:
            confidence -= 0.20
            warnings.append("OCR used")
        if self.wrapped:
            confidence -= 0.05
            warnings.append("Row was merged from wrapped lines")

        self.confidence = max(0.0, min(1.0, round(confidence, 3)))
        return {
            "source_pdf": self.source_pdf,
            "source_path": self.source_path,
            "page_number": self.page_number,
            "section": SECTION_LABEL,
            "category": CATEGORY_LABEL,
            "item_no": self.item_no,
            "qty": self.qty,
            "tag_no": self.tag_no,
            "size": self.size,
            "description": self.description,
            "material": self.material,
            "commodity_code": self.commodity_code,
            "remarks": self.remarks,
            "raw_text": self.raw_text,
            "bbox": json.dumps([round(v, 3) for v in self.bbox]),
            "confidence": self.confidence,
            "warnings": "; ".join(dict.fromkeys(warnings)),
        }


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def resolve_pdf_dir(requested: str | None) -> Path:
    if requested:
        return Path(requested)
    for candidate in (Path("6810 copy"), Path("6810 Copy")):
        if candidate.exists():
            return candidate
    return Path("6810 copy")


def list_pdfs(pdf_dir: Path, limit: int | None = None, include_old_revs: bool = False) -> list[Path]:
    pdfs = [
        path
        for path in sorted(pdf_dir.rglob("*.pdf"))
        if include_old_revs or "OLD REV_S" not in path.parts
    ]
    return pdfs[:limit] if limit else pdfs


def clean_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    metadata = metadata or {}
    return {key: str(metadata.get(key, "") or "") for key in METADATA_KEYS}


def box_values(box: Any) -> list[float]:
    if box is None:
        return []
    try:
        return [float(value) for value in box]
    except TypeError:
        return []


def cluster_words_into_lines(words: list[dict[str, Any]], y_tolerance: float = 3) -> list[Line]:
    if not words:
        return []

    lines: list[list[dict[str, Any]]] = []
    current = [words[0]]
    current_top = float(words[0]["top"])

    for word in sorted(words[1:], key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if abs(top - current_top) <= y_tolerance:
            current.append(word)
        else:
            lines.append(sorted(current, key=lambda item: float(item["x0"])))
            current = [word]
            current_top = top

    lines.append(sorted(current, key=lambda item: float(item["x0"])))
    return [Line(line) for line in lines]


def extract_words(page: Any) -> list[dict[str, Any]]:
    return page.extract_words(
        keep_blank_chars=False,
        use_text_flow=False,
        x_tolerance=2,
        y_tolerance=3,
    ) or []


def classify_text_layer(lines: list[Line], word_count: int) -> str:
    joined = normalize(" ".join(line.text for line in lines))
    has_anchor = any(
        anchor in joined
        for anchor in (
            "OTHERTHANSHOPMATERIALS",
            "INSTRUMENTS",
            "SHOPMATERIALS",
            "MATERIALS",
            "ITEM",
            "QTY",
            "DESCRIPTION",
        )
    )
    if word_count <= 10:
        return "IMAGE_ONLY"
    if word_count >= 50 and has_anchor:
        return "GOOD_TEXT_LAYER"
    return "POOR_TEXT_LAYER"


def is_column_header(line: Line) -> bool:
    norm = line.normalized
    return (
        ("DESCRIPTION" in norm)
        and ("QTY" in norm or "QUANTITY" in norm)
        and ("NO" in norm or "ITEM" in norm)
    )


def detect_header_columns(line: Line) -> dict[str, float]:
    columns: dict[str, float] = {}
    previous_text = ""
    for word in line.words:
        raw = str(word["text"]).strip()
        norm = normalize(raw)
        combined = normalize(previous_text + raw)
        column = HEADER_ALIASES.get(norm) or HEADER_ALIASES.get(combined)
        if column and column not in columns:
            columns[column] = float(word["x0"])
        previous_text = raw
    return columns


def build_column_ranges(columns: dict[str, float], page_width: float) -> dict[str, tuple[float, float]]:
    ordered = sorted(columns.items(), key=lambda item: item[1])
    ranges: dict[str, tuple[float, float]] = {}
    for idx, (name, xpos) in enumerate(ordered):
        left = 0.0 if idx == 0 else (ordered[idx - 1][1] + xpos) / 2
        right = page_width if idx == len(ordered) - 1 else (xpos + ordered[idx + 1][1]) / 2
        ranges[name] = (left, right)
    return ranges


def bucket_line(line: Line, ranges: dict[str, tuple[float, float]]) -> dict[str, str]:
    buckets = {name: [] for name in HEADER_ALIASES.values()}
    for word in line.words:
        x0 = float(word["x0"])
        text = str(word["text"]).strip()
        for name, (left, right) in ranges.items():
            if left <= x0 < right:
                buckets.setdefault(name, []).append(text)
                break
    return {key: " ".join(value).strip() for key, value in buckets.items()}


def is_section_header(line: Line) -> bool:
    return "OTHERTHANSHOPMATERIALS" in line.normalized


def is_instrument_header(line: Line) -> bool:
    norm = line.normalized
    return norm in INSTRUMENT_HEADERS or any(header in norm for header in INSTRUMENT_HEADERS if header != "INST")


def looks_like_row(line: Line) -> bool:
    tokens = [str(word["text"]).strip() for word in line.words]
    has_item = any(ITEM_RE.match(token) for token in tokens[:3])
    has_qty = any(QTY_RE.match(token) for token in tokens[-3:])
    return has_item and (has_qty or len(tokens) >= 3)


def is_category_header(line: Line) -> bool:
    norm = line.normalized
    if not norm:
        return False
    if norm.startswith("PIPESUPPORTS"):
        return True
    if looks_like_row(line):
        return False
    if norm in INSTRUMENT_HEADERS:
        return True
    if norm in CATEGORY_HEADERS:
        return True
    return any(norm.startswith(header) for header in CATEGORY_HEADERS if len(header) > 4)


def is_footer_or_title_line(line: Line) -> bool:
    norm = line.normalized
    return any(anchor in norm for anchor in FOOTER_OR_TITLE_ANCHORS)


def find_instrument_tag(text: str) -> str:
    match = INSTRUMENT_TAG_RE.search(text or "")
    if match:
        return match.group(0)

    compact = re.sub(r"\s+", "", text or "")
    match = COMPACT_INSTRUMENT_TAG_RE.search(compact)
    return match.group(0) if match else ""


def repair_spaced_digit_row(row: ParsedRow) -> None:
    tokens = row.raw_text.split()
    if len(tokens) < 8 or not row.tag_no:
        return

    if all(token.isdigit() for token in tokens[:3]):
        row.item_no = tokens[0] + tokens[1]
        row.size = tokens[2]
        row.qty = tokens[-1] if QTY_RE.match(tokens[-1]) else row.qty
        row.description = row.tag_no
        row.commodity_code = row.tag_no


def repair_tag_row_from_raw(row: ParsedRow) -> None:
    if not row.tag_no or " " in row.tag_no:
        return

    size_pattern = r'\d+(?:X\d+(?:\.\d+/\d+|/\d+|\.\d+)?)?(?:\.\d+/\d+|/\d+|\.\d+)?"?'
    tag_pattern = re.escape(row.tag_no)
    pattern = re.compile(
        rf"(?P<item>\d{{1,3}})\s+(?P<size>{size_pattern})\s+"
        rf"(?P<tag>{tag_pattern})\s+(?P<commodity>{tag_pattern})\s+(?P<qty>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(row.raw_text))
    if not matches:
        return

    match = matches[-1]
    row.item_no = match.group("item")
    row.size = match.group("size")
    row.qty = match.group("qty")
    row.description = match.group("tag")
    row.commodity_code = match.group("commodity")


def looks_like_material_row(line: Line, ranges: dict[str, tuple[float, float]] | None) -> bool:
    if is_footer_or_title_line(line):
        return False
    if is_category_header(line):
        return False

    if ranges:
        buckets = bucket_line(line, ranges)
        if first_item_value(buckets.get("item_no", "")) and first_qty_value(buckets.get("qty", "")):
            return True

    tokens = [str(word["text"]).strip() for word in line.words if str(word["text"]).strip()]
    if len(tokens) < 3:
        return False
    return bool(ITEM_RE.match(tokens[0]) and any(QTY_RE.match(token) for token in tokens[-3:]))


def merge_bbox(first: list[float], second: list[float]) -> list[float]:
    if not first:
        return second
    if not second:
        return first
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


def infer_row_without_columns(line: Line, source_pdf: str, source_path: str, page_number: int) -> ParsedRow:
    tokens = [str(word["text"]).strip() for word in line.words if str(word["text"]).strip()]
    row = ParsedRow(
        source_pdf=source_pdf,
        source_path=source_path,
        page_number=page_number,
        raw_text=line.text,
        bbox=line.bbox,
        column_positions_uncertain=True,
    )
    if tokens and ITEM_RE.match(tokens[0]):
        row.item_no = tokens[0]
        tokens = tokens[1:]
    if tokens and SIZE_RE.match(tokens[0]):
        row.size = tokens[0]
        tokens = tokens[1:]
    if tokens and QTY_RE.match(tokens[-1]):
        row.qty = tokens[-1]
        tokens = tokens[:-1]

    row.description = " ".join(tokens).strip()
    row.tag_no = find_instrument_tag(row.raw_text)
    return row


def parse_new_row(
    line: Line,
    ranges: dict[str, tuple[float, float]] | None,
    source_pdf: str,
    source_path: str,
    page_number: int,
) -> ParsedRow:
    if not ranges:
        return infer_row_without_columns(line, source_pdf, source_path, page_number)

    buckets = bucket_line(line, ranges)
    row = ParsedRow(
        source_pdf=source_pdf,
        source_path=source_path,
        page_number=page_number,
        item_no=first_item_value(buckets.get("item_no", "")),
        qty=first_qty_value(buckets.get("qty", "")),
        size=first_size_value(buckets.get("size", "")),
        description=buckets.get("description", "").strip(),
        material=buckets.get("material", "").strip(),
        commodity_code=buckets.get("commodity_code", "").strip(),
        remarks=buckets.get("remarks", "").strip(),
        raw_text=line.text,
        bbox=line.bbox,
    )

    if not row.description:
        row.description = " ".join(
            value
            for key, value in buckets.items()
            if key not in {"item_no", "qty", "size"} and value
        ).strip()

    row.tag_no = (
        find_instrument_tag(row.commodity_code)
        or find_instrument_tag(row.description)
        or find_instrument_tag(row.raw_text)
    )
    if row.tag_no and (not row.description or normalize(row.tag_no) in normalize(row.description)):
        row.description = row.tag_no
    repair_tag_row_from_raw(row)
    repair_spaced_digit_row(row)

    return row


def first_item_value(text: str) -> str:
    return next((token for token in text.split() if ITEM_RE.match(token)), "").strip()


def first_qty_value(text: str) -> str:
    return next((token for token in reversed(text.split()) if QTY_RE.match(token)), "").strip()


def first_size_value(text: str) -> str:
    spaced_fraction = re.search(r'\d+(?:\.\d+)?\s*/\s*\d+"?', text)
    if spaced_fraction:
        return re.sub(r"\s+", "", spaced_fraction.group(0)).strip()
    return next((token for token in text.split() if SIZE_RE.match(token)), "").strip()


def append_continuation(row: ParsedRow, line: Line, ranges: dict[str, tuple[float, float]] | None) -> None:
    row.raw_text = f"{row.raw_text} {line.text}".strip()
    row.bbox = merge_bbox(row.bbox, line.bbox)
    row.wrapped = True

    if ranges:
        buckets = bucket_line(line, ranges)
        continuation = " ".join(
            part
            for part in (
                buckets.get("description", ""),
                buckets.get("material", ""),
                buckets.get("commodity_code", ""),
                buckets.get("remarks", ""),
            )
            if part
        ).strip()
    else:
        continuation = line.text

    if continuation:
        row.description = f"{row.description} {continuation}".strip()

    if not row.qty and ranges:
        row.qty = first_qty_value(bucket_line(line, ranges).get("qty", ""))

    if not row.tag_no:
            row.tag_no = find_instrument_tag(row.raw_text)


def should_append_continuation(row: ParsedRow, line: Line) -> bool:
    if is_footer_or_title_line(line):
        return False
    if find_instrument_tag(line.text):
        return True
    if row.tag_no:
        norm = line.normalized
        return norm.startswith("CONNTO") or norm.startswith("CONNECTIONTO")
    return bool(line.text and not is_category_header(line))


def is_plausible_candidate(row: ParsedRow) -> bool:
    return bool(row.qty or row.tag_no or row.description or row.commodity_code)


def extract_rows_from_page(
    lines: list[Line],
    source_pdf: str,
    source_path: str,
    page_number: int,
    page_width: float,
    page_height: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    found_section = False
    found_instruments = False
    section_count = 0
    instruments_count = 0
    warnings: list[str] = []

    ranges: dict[str, tuple[float, float]] | None = None
    table_left = page_width * 0.60
    table_right = page_width
    in_target_section = False
    in_instruments = False
    current: ParsedRow | None = None

    def flush_current() -> None:
        nonlocal current
        if current is not None:
            if is_plausible_candidate(current):
                rows.append(current.finalize())
            current = None

    for source_line in lines:
        table_words = [
            word
            for word in source_line.words
            if float(word["x0"]) >= table_left - 5
        ]
        if not table_words:
            continue
        line = Line(table_words)

        if is_column_header(line):
            columns = detect_header_columns(line)
            if {"item_no", "qty", "description"}.issubset(columns):
                ranges = build_column_ranges(columns, page_width)
                table_left = min(table_left, line.bbox[0])
                table_right = max(table_right, line.bbox[2])
            continue

        if is_section_header(line):
            flush_current()
            found_section = True
            section_count += 1
            in_target_section = True
            in_instruments = False
            table_left = min(table_left, line.bbox[0])
            table_right = max(table_right, line.bbox[2])
            continue

        if not in_target_section:
            continue

        if "PIECEMARKS" in normalize(line.text):
            flush_current()
            in_target_section = False
            in_instruments = False
            continue

        if is_instrument_header(line):
            flush_current()
            found_instruments = True
            instruments_count += 1
            in_instruments = True
            continue

        if is_footer_or_title_line(line) and in_instruments:
            flush_current()
            in_target_section = False
            in_instruments = False
            continue

        if is_category_header(line):
            flush_current()
            in_instruments = False
            continue

        if not in_instruments:
            continue

        if not line.text:
            continue

        if looks_like_material_row(line, ranges):
            flush_current()
            current = parse_new_row(line, ranges, source_pdf, source_path, page_number)
            if ranges is None:
                current.column_positions_uncertain = True
            if line.bbox[0] < table_left - 10 or line.bbox[2] > table_right + 50:
                current.category_boundary_unclear = True
            continue

        if current is not None and should_append_continuation(current, line):
            append_continuation(current, line, ranges)

    flush_current()

    if section_count > 1:
        warnings.append("Multiple OTHER THAN SHOP MATERIALS sections found")
    if instruments_count > 1:
        warnings.append("Multiple INSTRUMENTS sections found")

    return rows, {
        "found_section": found_section,
        "found_instruments": found_instruments,
        "section_count": section_count,
        "instruments_count": instruments_count,
        "warnings": warnings,
    }


def page_metadata(page: Any, page_number: int, word_count: int, has_text_layer: bool) -> dict[str, Any]:
    mediabox = box_values(getattr(page, "mediabox", None))
    cropbox = box_values(getattr(page, "cropbox", None))
    return {
        "page_number": page_number,
        "width": float(page.width),
        "height": float(page.height),
        "rotation": int(getattr(page, "rotation", 0) or 0),
        "mediabox": mediabox,
        "cropbox": cropbox,
        "word_count": word_count,
        "has_text_layer": has_text_layer,
    }


def extract_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdf_rows: list[dict[str, Any]] = []
    source_path = str(pdf_path)
    source_pdf = pdf_path.name
    audit_warnings: list[str] = []
    page_records: list[dict[str, Any]] = []
    page_statuses: list[str] = []
    section_found = False
    instruments_found = False

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            metadata = clean_metadata(pdf.metadata)
            for page_index, page in enumerate(pdf.pages, start=1):
                words = extract_words(page)
                lines = cluster_words_into_lines(sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))))
                status = classify_text_layer(lines, len(words))
                page_statuses.append(status)
                page_records.append(page_metadata(page, page_index, len(words), bool(words)))

                if getattr(page, "rotation", 0):
                    audit_warnings.append(f"Page {page_index}: page rotation detected")
                if page_records[-1]["mediabox"] and page_records[-1]["cropbox"] and page_records[-1]["mediabox"] != page_records[-1]["cropbox"]:
                    audit_warnings.append(f"Page {page_index}: CropBox differs from MediaBox")
                if status == "IMAGE_ONLY":
                    audit_warnings.append(f"Page {page_index}: PDF appears image-only")
                elif status == "POOR_TEXT_LAYER":
                    audit_warnings.append(f"Page {page_index}: poor text layer")

                page_rows, page_info = extract_rows_from_page(
                    lines,
                    source_pdf,
                    source_path,
                    page_index,
                    float(page.width),
                    float(page.height),
                )
                pdf_rows.extend(page_rows)
                section_found = section_found or page_info["found_section"]
                instruments_found = instruments_found or page_info["found_instruments"]
                audit_warnings.extend(page_info["warnings"])
    except Exception as exc:
        return [], {
            "source_pdf": source_pdf,
            "source_path": source_path,
            "page_count": 0,
            "pages_checked": 0,
            "text_layer_status": "ERROR",
            "other_than_shop_materials_found": False,
            "instruments_category_found": False,
            "instrument_rows_found": 0,
            "metadata": {},
            "page_metadata": [],
            "warnings": [repr(exc)],
        }

    if not section_found:
        audit_warnings.append("No OTHER THAN SHOP MATERIALS section found")
    elif not instruments_found:
        audit_warnings.append("OTHER THAN SHOP MATERIALS found but INSTRUMENTS not found")
    elif not pdf_rows:
        audit_warnings.append("INSTRUMENTS found but no rows underneath")

    audit = {
        "source_pdf": source_pdf,
        "source_path": source_path,
        "page_count": len(page_records),
        "pages_checked": len(page_records),
        "text_layer_status": combine_text_statuses(page_statuses),
        "other_than_shop_materials_found": section_found,
        "instruments_category_found": instruments_found,
        "instrument_rows_found": len(pdf_rows),
        "metadata": metadata,
        "page_metadata": page_records,
        "warnings": list(dict.fromkeys(audit_warnings)),
    }
    return pdf_rows, audit


def combine_text_statuses(statuses: list[str]) -> str:
    if not statuses:
        return "IMAGE_ONLY"
    if all(status == "GOOD_TEXT_LAYER" for status in statuses):
        return "GOOD_TEXT_LAYER"
    if all(status == "IMAGE_ONLY" for status in statuses):
        return "IMAGE_ONLY"
    return "POOR_TEXT_LAYER"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ROW_COLUMNS})


def run(pdf_dir: Path, output_dir: Path, limit: int | None = None, include_old_revs: bool = False) -> dict[str, Any]:
    pdfs = list_pdfs(pdf_dir, limit, include_old_revs=include_old_revs)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for index, pdf_path in enumerate(pdfs, start=1):
        rows, audit = extract_pdf(pdf_path)
        raw_rows.extend(rows)
        audits.append(audit)
        if index % 100 == 0:
            print(f"Processed {index}/{len(pdfs)} PDFs | rows: {len(raw_rows)}")

    clean_rows = [row for row in raw_rows if float(row["confidence"]) >= 0.65]
    review_rows = [row for row in raw_rows if float(row["confidence"]) < 0.65]

    write_csv(output_dir / "instruments_raw.csv", raw_rows)
    write_csv(output_dir / "instruments_clean.csv", clean_rows)
    write_csv(output_dir / "low_confidence_review.csv", review_rows)
    (output_dir / "pdf_extraction_audit.json").write_text(
        json.dumps(audits, indent=2),
        encoding="utf-8",
    )

    return {
        "pdfs_scanned": len(pdfs),
        "raw_rows": len(raw_rows),
        "clean_rows": len(clean_rows),
        "review_rows": len(review_rows),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", default=None, help="PDF directory. Defaults to '6810 copy' or '6810 Copy'.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV/JSON deliverables.")
    parser.add_argument("--limit", type=int, default=None, help="Optional PDF limit for smoke tests.")
    parser.add_argument(
        "--include-old-revs",
        action="store_true",
        help="Include PDFs under OLD REV_S. Default keeps the original exclusion rule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        resolve_pdf_dir(args.pdf_dir),
        Path(args.output_dir),
        args.limit,
        include_old_revs=args.include_old_revs,
    )
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
