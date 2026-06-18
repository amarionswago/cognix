from pathlib import Path
from types import SimpleNamespace

from app.config import get_settings
from app.database import db_session, init_db
from app.services import ocr, parsers
from app.services.ingest import run_ingest
from app.services.ocr import OCRResult, estimate_ocr_confidence, image_data_url, ocr_image_result, ocr_pdf


def test_image_data_url_encodes_local_image(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    path.write_bytes(b"\x89PNG\r\n")

    data_url = image_data_url(path)

    assert data_url.startswith("data:image/png;base64,")
    assert data_url.endswith("iVBORw0K")


def test_ocr_confidence_scores_text_quality() -> None:
    good = estimate_ocr_confidence("Invoice Total 42 USD due on 2026-06-17")
    empty = estimate_ocr_confidence("[Image OCR produced no text]")

    assert good > 0.5
    assert empty == 0.0


def test_image_ocr_result_preserves_method_confidence_and_metadata(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "scan.png"
    path.write_bytes(b"\x89PNG\r\n")

    monkeypatch.setattr(ocr.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "tesseract" else None)
    monkeypatch.setattr(
        ocr.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Receipt total 12.50 USD", stderr="", returncode=0),
    )

    result = ocr_image_result(path)

    assert result.method == "image-ocr-tesseract"
    assert result.confidence > 0.4
    assert result.metadata["characters"] == len("Receipt total 12.50 USD")


def test_pdf_ocr_assembles_page_text_with_confidence(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4 scanned fixture")

    def fake_which(name: str) -> str | None:
        if name in {"pdftoppm", "tesseract"}:
            return f"/usr/bin/{name}"
        return None

    def fake_run(args, **kwargs):
        executable = Path(args[0]).name
        if executable == "pdftoppm":
            output_prefix = Path(args[-1])
            output_prefix.parent.mkdir(parents=True, exist_ok=True)
            (output_prefix.parent / "page-1.png").write_bytes(b"page1")
            (output_prefix.parent / "page-2.png").write_bytes(b"page2")
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        image_name = Path(args[1]).name
        return SimpleNamespace(stdout=f"OCR text from {image_name}", stderr="", returncode=0)

    monkeypatch.setattr(ocr.shutil, "which", fake_which)
    monkeypatch.setattr(ocr.subprocess, "run", fake_run)

    result = ocr_pdf(path)

    assert result.method == "pdf-ocr-tesseract"
    assert result.page_count == 2
    assert result.confidence > 0.3
    assert "--- OCR Page 1 ---" in result.text
    assert "OCR text from page-2.png" in result.text


def test_ingest_persists_ocr_extraction_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIX_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("COGNIX_DATABASE_PATH", str(tmp_path / "library.sqlite"))
    monkeypatch.setenv("COGNIX_CHROMA_ENABLED", "false")
    get_settings.cache_clear()
    init_db()

    raw_dir = get_settings().resolved_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_path = raw_dir / "whiteboard.png"
    image_path.write_bytes(b"\x89PNG\r\n")

    monkeypatch.setattr(
        parsers,
        "ocr_image_result",
        lambda _path: OCRResult(
            text="Whiteboard roadmap says retrieval, OCR, and calibration.",
            method="image-ocr-test",
            confidence=0.88,
            page_count=1,
            warnings=(),
            metadata={"fixture": True},
        ),
    )

    result = run_ingest("test")

    with db_session() as conn:
        artifact = conn.execute("SELECT method, confidence, page_count, text_length FROM extraction_artifacts").fetchone()
        chunk = conn.execute("SELECT text FROM chunks LIMIT 1").fetchone()

    assert result["processed"] == 1
    assert artifact["method"] == "image-ocr-test"
    assert artifact["confidence"] == 0.88
    assert artifact["page_count"] == 1
    assert artifact["text_length"] > 0
    assert "Whiteboard roadmap" in chunk["text"]

    get_settings.cache_clear()
