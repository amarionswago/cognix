import csv
import html
import json
import re
from html.parser import HTMLParser
from dataclasses import dataclass
from pathlib import Path

from app.services.adapters import adapt_known_export
from app.services.ocr import OCRResult, ocr_image_result, ocr_metadata_json, ocr_pdf


PARSER_VERSION = "parser-v4"
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".htm", ".json", ".csv", ".yml", ".yaml", ".toml"}


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    text: str
    method: str
    extraction_confidence: float = 1.0
    page_count: int = 0
    warnings: tuple[str, ...] = ()
    metadata_json: str = "{}"


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
        parsed = _parse_pdf_document(path)
        adapted = adapt_known_export(path, parsed.text)
        return ParsedDocument(
            title=path.stem,
            text=adapted[0],
            method=adapted[1],
            extraction_confidence=parsed.extraction_confidence,
            page_count=parsed.page_count,
            warnings=parsed.warnings,
            metadata_json=parsed.metadata_json,
        ) if adapted else parsed
    if extension in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".tiff", ".bmp"}:
        result = ocr_image_result(path)
        return _document_from_ocr(path, result)
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
    parser = ReadableHTMLParser()
    parser.feed(raw)
    parser.close()
    return parser.text(path.name)


class ReadableHTMLParser(HTMLParser):
    """Extract human-readable page content without dumping scripts or SEO noise."""

    IGNORED_TAGS = {"script", "style", "svg", "canvas", "template"}
    BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self.meta_description = ""
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = {name.lower(): value or "" for name, value in attrs}
        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "meta":
            name = attr_map.get("name", "").lower()
            prop = attr_map.get("property", "").lower()
            if name == "description" or prop == "og:description":
                self.meta_description = attr_map.get("content", "").strip()
            return
        if self._ignored_depth:
            return
        if tag == "img" and attr_map.get("alt"):
            self.body_parts.append(f" Image: {attr_map['alt'].strip()} ")
        elif tag in self.BLOCK_TAGS:
            self.body_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if not self._ignored_depth and tag in self.BLOCK_TAGS:
            self.body_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        value = data.strip()
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        else:
            self.body_parts.append(value)

    def text(self, fallback_title: str) -> str:
        title = clean_extracted_text(" ".join(self.title_parts)) or fallback_title
        description = clean_extracted_text(self.meta_description)
        body = clean_extracted_text(" ".join(self.body_parts))
        sections = [f"HTML title: {title}"]
        if description and description.lower() != title.lower():
            sections.append(f"HTML description: {description}")
        if body and body.lower() != title.lower():
            sections.append(body)
        return "\n\n".join(sections).strip()


def clean_extracted_text(text: str) -> str:
    text = html.unescape(text)
    lines = []
    for line in re.split(r"[\r\n]+", text):
        compact = re.sub(r"\s+", " ", line).strip()
        if compact:
            lines.append(compact)
    return "\n".join(lines)


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


def _parse_pdf_document(path: Path) -> ParsedDocument:
    text = _parse_pdf(path)
    if text == "[PDF contained no selectable text. OCR is required.]":
        result = ocr_pdf(path)
        return _document_from_ocr(path, result)
    method = "pdf-text"
    confidence = 0.95
    warnings: tuple[str, ...] = ()
    if text.startswith("[PDF parser fallback:"):
        method = "pdf-text-fallback"
        confidence = 0.55
        warnings = ("PDF parser could not read a valid container; recovered raw text directly.",)
    elif text.startswith("[PDF parser dependency"):
        confidence = 0.0
        warnings = ("pypdf is not installed.",)
    return ParsedDocument(
        title=path.stem,
        text=text,
        method=method,
        extraction_confidence=confidence,
        page_count=_count_pdf_pages_from_text(text),
        warnings=warnings,
        metadata_json=json.dumps(
            {
                "method": method,
                "confidence": confidence,
                "page_count": _count_pdf_pages_from_text(text),
                "warnings": list(warnings),
            },
            sort_keys=True,
        ),
    )


def _document_from_ocr(path: Path, result: OCRResult) -> ParsedDocument:
    return ParsedDocument(
        title=path.stem,
        text=result.text,
        method=result.method,
        extraction_confidence=result.confidence,
        page_count=result.page_count,
        warnings=result.warnings,
        metadata_json=ocr_metadata_json(result),
    )


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[PDF parser dependency pypdf is not installed.]"

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        if _looks_like_text(path):
            return f"[PDF parser fallback: file was not a valid PDF container; recovered text directly. Error: {type(exc).__name__}]\n\n{_read_text(path)}"
        raise
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"\n\n--- Page {index} ---\n{page_text}")
    text = "".join(pages).strip()
    return text or "[PDF contained no selectable text. OCR is required.]"


def _count_pdf_pages_from_text(text: str) -> int:
    return len(re.findall(r"--- Page \d+ ---", text))
