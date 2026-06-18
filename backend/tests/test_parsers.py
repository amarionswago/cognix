from pathlib import Path

from app.services.parsers import parse_file


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_html_parser_extracts_readable_content_without_script_or_keywords(tmp_path: Path) -> None:
    path = write(
        tmp_path / "ai_research.html",
        """
        <!doctype html>
        <html>
          <head>
            <title>Artificial Intelligence Research Notes</title>
            <meta name="keywords" content="spam, dump, unrelated">
            <meta name="description" content="A concise page about neural systems.">
            <script>window.secret = "do not index this";</script>
            <style>.hidden { display: none; }</style>
          </head>
          <body>
            <main>
              <h1>Artificial intelligence</h1>
              <p>Neural networks learn useful representations from data.</p>
              <img alt="Transformer attention diagram">
            </main>
          </body>
        </html>
        """,
    )

    parsed = parse_file(path)

    assert parsed.method == "html-text"
    assert "HTML title: Artificial Intelligence Research Notes" in parsed.text
    assert "HTML description: A concise page about neural systems." in parsed.text
    assert "Neural networks learn useful representations from data." in parsed.text
    assert "Transformer attention diagram" in parsed.text
    assert "do not index this" not in parsed.text
    assert "spam, dump, unrelated" not in parsed.text


def test_common_supported_formats_do_not_use_unsupported_placeholder(tmp_path: Path) -> None:
    files = [
        write(tmp_path / "note.txt", "plain text note"),
        write(tmp_path / "note.md", "# Markdown note"),
        write(tmp_path / "data.json", '{"topic": "ai"}'),
        write(tmp_path / "rows.csv", "topic,value\nai,1\n"),
        write(tmp_path / "message.eml", "Subject: Test\n\nEmail body"),
        write(tmp_path / "feed.xml", "<root><item>XML body</item></root>"),
        write(tmp_path / "trace.log", "log line"),
        write(tmp_path / "script.py", "print('hello')"),
    ]

    for path in files:
        parsed = parse_file(path)
        assert parsed.text
        assert parsed.method != "unsupported-placeholder", path.name
        assert "Unsupported file type" not in parsed.text
