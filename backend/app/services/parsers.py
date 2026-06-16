import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.services.adapters import adapt_known_export
from app.services.ocr import ocr_image


PARSER_VERSION = "parser-v2"
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".htm", ".json", ".csv", ".yml", ".yaml", ".toml"}


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    text: str
    method: str


def parse_file(path: Path) -> ParsedDocument:
    extension = path.suffix.lower()
    if not extension and _looks_like_text(path):
        return ParsedDocument(title=path.name, text=_read_text(path), method="extensionless-text")
    if extension in {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".yml", ".yaml", ".toml"}:
        return ParsedDocument(title=path.stem, text=_read_text(path), method="plain-text")
    if extension in {".html", ".htm"}:
        return ParsedDocument(title=path.stem, text=_parse_html(path), method="html-text")
    if extension == ".json":
        return ParsedDocument(title=path.stem, text=_parse_json(path), method="json-text")
    if extension == ".csv":
        return ParsedDocument(title=path.stem, text=_parse_csv(path), method="csv-summary")
    if extension == ".pdf":
        parsed = ParsedDocument(title=path.stem, text=_parse_pdf(path), method="pdf-text")
        adapted = adapt_known_export(path, parsed.text)
        return ParsedDocument(path.stem, adapted[0], adapted[1]) if adapted else parsed
    if extension in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".tiff", ".bmp"}:
        text, method = ocr_image(path)
        return ParsedDocument(title=path.stem, text=text, method=method)
    parsed = _parse_supported_text(path, extension)
    if parsed:
        adapted = adapt_known_export(path, parsed.text)
        return ParsedDocument(path.stem, adapted[0], adapted[1]) if adapted else parsed
    return ParsedDocument(title=path.stem, text=f"[Unsupported file type in version 1: {path.suffix}]", method="unsupported-placeholder")


def _looks_like_text(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        try:
            sample.decode("latin-1")
            return True
        except UnicodeDecodeError:
            return False


def _parse_supported_text(path: Path, extension: str) -> ParsedDocument | None:
    if extension in {".eml", ".mbox", ".xml", ".log"}:
        return ParsedDocument(title=path.stem, text=_read_text(path), method=f"{extension[1:]}-text")
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def _parse_html(path: Path) -> str:
    raw = _read_text(path)
    without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _parse_json(path: Path) -> str:
    raw = _read_text(path)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(data, indent=2, ensure_ascii=True)


def _parse_csv(path: Path) -> str:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for idx, row in enumerate(reader):
            if idx >= 200:
                break
            rows.append(dict(row))
    lines = [f"CSV file: {path.name}", f"Columns: {', '.join(headers)}", f"Preview rows: {len(rows)}"]
    for idx, row in enumerate(rows[:50], start=1):
        compact = "; ".join(f"{key}={value}" for key, value in row.items())
        lines.append(f"Row {idx}: {compact}")
    return "\n".join(lines)


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[PDF parser dependency pypdf is not installed.]"

    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(f"\n\n--- Page {index} ---\n{page_text}")
    text = "".join(pages).strip()
    return text or "[PDF contained no selectable text. OCR is required.]"
