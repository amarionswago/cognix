import shutil
import subprocess
from pathlib import Path


def ocr_image(path: Path) -> tuple[str, str]:
    """Run local OCR if a supported command-line OCR tool is available."""
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return (
            f"[Image tracked but OCR tool is not installed: {path.name}]",
            "image-ocr-missing-tesseract",
        )
    try:
        result = subprocess.run(
            [tesseract, str(path), "stdout"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = result.stdout.strip()
        if text:
            return text, "image-ocr-tesseract"
        return f"[Image OCR produced no text: {path.name}]", "image-ocr-empty"
    except Exception as exc:
        return f"[Image OCR failed for {path.name}: {exc}]", "image-ocr-error"
