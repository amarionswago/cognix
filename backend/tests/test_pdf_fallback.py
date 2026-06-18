from pathlib import Path

from app.services.parsers import parse_file


def test_text_recoverable_malformed_pdf_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    path.write_text("Plain text trapped in a malformed PDF extension.", encoding="utf-8")

    parsed = parse_file(path)

    assert "recovered text directly" in parsed.text
    assert "Plain text trapped" in parsed.text
