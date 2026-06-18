import base64
import json
import mimetypes
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import get_settings
from app.services.providers import get_provider_row, parse_openai_response, resolve_provider_key


@dataclass(frozen=True)
class OCRResult:
    """Structured OCR/vision extraction result."""

    text: str
    method: str
    confidence: float
    page_count: int = 1
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


def ocr_image(path: Path) -> tuple[str, str]:
    """Run OCR for a local image and return the legacy text/method tuple."""
    result = ocr_image_result(path)
    return result.text, result.method


def ocr_image_result(path: Path) -> OCRResult:
    """Run local/cloud OCR for a supported image and preserve extraction metadata."""
    settings = get_settings()
    if settings.vision_backend.strip().lower() in {"openai", "cloud", "cloud-vision", "llm-vision"}:
        result = describe_image_with_openai(path)
        if result.text:
            return result
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return OCRResult(
            text=f"[Image tracked but OCR tool is not installed: {path.name}]",
            method="image-ocr-missing-tesseract",
            confidence=0.0,
            warnings=("Install tesseract to extract local image text.",),
            metadata={"path": path.name},
        )
    return _run_tesseract_on_image(path, "image-ocr-tesseract")


def ocr_pdf(path: Path) -> OCRResult:
    """OCR a scanned/image-only PDF with local command-line tools when available."""
    pdftoppm = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if not pdftoppm or not tesseract:
        missing = []
        if not pdftoppm:
            missing.append("pdftoppm")
        if not tesseract:
            missing.append("tesseract")
        return OCRResult(
            text=f"[PDF contained no selectable text. OCR tools missing: {', '.join(missing)}.]",
            method="pdf-ocr-missing-tools",
            confidence=0.0,
            page_count=0,
            warnings=(f"Install {', '.join(missing)} to OCR scanned PDFs.",),
            metadata={"missing_tools": ",".join(missing), "path": path.name},
        )
    with tempfile.TemporaryDirectory(prefix="cognix-pdf-ocr-") as temp_dir:
        output_prefix = Path(temp_dir) / "page"
        try:
            convert_result = subprocess.run(
                [pdftoppm, "-png", "-r", "200", str(path), str(output_prefix)],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except Exception as exc:
            return OCRResult(
                text=f"[PDF OCR conversion failed for {path.name}: {exc}]",
                method="pdf-ocr-conversion-error",
                confidence=0.0,
                page_count=0,
                warnings=(str(exc),),
                metadata={"path": path.name},
            )
        if convert_result.returncode != 0:
            message = (convert_result.stderr or convert_result.stdout or "unknown conversion error").strip()
            return OCRResult(
                text=f"[PDF OCR conversion failed for {path.name}: {message}]",
                method="pdf-ocr-conversion-failed",
                confidence=0.0,
                page_count=0,
                warnings=(message,),
                metadata={"path": path.name, "returncode": convert_result.returncode},
            )
        page_images = sorted(Path(temp_dir).glob("page-*.png"))
        if not page_images:
            return OCRResult(
                text=f"[PDF OCR conversion produced no page images: {path.name}]",
                method="pdf-ocr-no-pages",
                confidence=0.0,
                page_count=0,
                warnings=("pdftoppm produced no page images.",),
                metadata={"path": path.name},
            )
        page_results = [_run_tesseract_on_image(page, "pdf-page-ocr-tesseract") for page in page_images]
    sections = []
    warnings = []
    for index, result in enumerate(page_results, start=1):
        if result.warnings:
            warnings.extend(result.warnings)
        sections.append(f"--- OCR Page {index} ---\n{result.text}")
    combined_text = "\n\n".join(sections).strip()
    confidence = sum(result.confidence for result in page_results) / max(1, len(page_results))
    method = "pdf-ocr-tesseract" if combined_text else "pdf-ocr-empty"
    return OCRResult(
        text=combined_text or f"[PDF OCR produced no text: {path.name}]",
        method=method,
        confidence=round(confidence, 4),
        page_count=len(page_results),
        warnings=tuple(warnings),
        metadata={"path": path.name, "pages": len(page_results)},
    )


def _run_tesseract_on_image(path: Path, method: str) -> OCRResult:
    """Run Tesseract on a single image-like file."""
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return OCRResult(
            text=f"[Image tracked but OCR tool is not installed: {path.name}]",
            method="image-ocr-missing-tesseract",
            confidence=0.0,
            warnings=("Install tesseract to extract local image text.",),
            metadata={"path": path.name},
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
            return OCRResult(
                text=text,
                method=method,
                confidence=estimate_ocr_confidence(text),
                metadata={"path": path.name, "characters": len(text)},
            )
        return OCRResult(
            text=f"[Image OCR produced no text: {path.name}]",
            method="image-ocr-empty" if method.startswith("image") else "pdf-page-ocr-empty",
            confidence=0.0,
            warnings=("OCR completed but produced no text.",),
            metadata={"path": path.name},
        )
    except Exception as exc:
        return OCRResult(
            text=f"[Image OCR failed for {path.name}: {exc}]",
            method="image-ocr-error" if method.startswith("image") else "pdf-page-ocr-error",
            confidence=0.0,
            warnings=(str(exc),),
            metadata={"path": path.name},
        )


def describe_image_with_openai(path: Path) -> OCRResult:
    """Describe an image through OpenAI Responses vision when configured."""
    key = resolve_provider_key("openai", get_provider_row("openai"))
    if not key.value:
        return OCRResult(
            text="",
            method="image-vision-missing-openai-key",
            confidence=0.0,
            warnings=("OpenAI vision selected but no OpenAI key is configured.",),
        )
    settings = get_settings()
    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {key.value}", "Content-Type": "application/json"},
            json={
                "model": settings.openai_vision_model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Extract all visible text from this image, then summarize the visual context "
                                    "for a searchable knowledge library. Preserve labels, table text, headings, "
                                    "and document-like structure when visible."
                                ),
                            },
                            {"type": "input_image", "image_url": image_data_url(path), "detail": "high"},
                        ],
                    }
                ],
            },
            timeout=90,
        )
    except Exception as exc:
        return OCRResult(
            text=f"[Image vision failed for {path.name}: {exc}]",
            method="image-vision-error",
            confidence=0.0,
            warnings=(str(exc),),
            metadata={"path": path.name},
        )
    text = parse_openai_response(response)
    if not text:
        return OCRResult(
            text="",
            method="image-vision-empty",
            confidence=0.0,
            warnings=("OpenAI vision response contained no extracted text.",),
            metadata={"path": path.name},
        )
    extracted = f"Vision extraction for {path.name}:\n{text.strip()}"
    return OCRResult(
        text=extracted,
        method="image-vision-openai",
        confidence=estimate_ocr_confidence(extracted),
        metadata={"path": path.name, "model": settings.openai_vision_model},
    )


def image_data_url(path: Path) -> str:
    """Return a base64 data URL for a local image file."""
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def estimate_ocr_confidence(text: str) -> float:
    """Estimate OCR quality from extracted text shape.

    This is not Tesseract's internal confidence. It is a lightweight quality
    score that catches empty/noisy extraction and gives downstream health checks
    a stable signal before deeper OCR engines are added.
    """
    cleaned = text.strip()
    if not cleaned or cleaned.startswith("["):
        return 0.0
    characters = len(cleaned)
    alnum = sum(1 for char in cleaned if char.isalnum())
    whitespace = sum(1 for char in cleaned if char.isspace())
    alnum_ratio = alnum / max(1, characters)
    word_count = len(cleaned.split())
    length_score = min(1.0, word_count / 30)
    noise_penalty = max(0.0, 1.0 - abs(whitespace / max(1, characters) - 0.15))
    return round(max(0.05, min(0.98, (alnum_ratio * 0.65 + length_score * 0.25 + noise_penalty * 0.10))), 4)


def ocr_metadata_json(result: OCRResult) -> str:
    """Serialize OCR metadata for database storage."""
    return json.dumps(
        {
            "method": result.method,
            "confidence": result.confidence,
            "page_count": result.page_count,
            "warnings": list(result.warnings),
            "metadata": result.metadata,
        },
        sort_keys=True,
    )
