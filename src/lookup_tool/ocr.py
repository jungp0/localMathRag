from __future__ import annotations

from pathlib import Path
from typing import Any


def try_ocr_image(path: str | Path) -> tuple[str | None, dict[str, Any]]:
    """Best-effort OCR hook.

    The project keeps OCR optional so the base tool runs without heavy runtime
    dependencies. If PaddleOCR is installed, this returns recognized text;
    otherwise it is a silent no-op.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return None, {"ocr_provider": None, "ocr_status": "unavailable"}

    try:
        engine = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        result = engine.predict(str(path))
    except Exception as exc:
        return None, {"ocr_provider": "paddleocr", "ocr_status": "error", "ocr_error": str(exc)}

    text_lines: list[str] = []
    for page in result if isinstance(result, list) else [result]:
        if isinstance(page, dict):
            for key in ("rec_texts", "texts"):
                value = page.get(key)
                if isinstance(value, list):
                    text_lines.extend(str(item) for item in value if str(item).strip())
        elif hasattr(page, "json"):
            try:
                payload = page.json
                if isinstance(payload, dict):
                    value = payload.get("rec_texts") or payload.get("texts")
                    if isinstance(value, list):
                        text_lines.extend(str(item) for item in value if str(item).strip())
            except Exception:
                pass
    text = "\n".join(text_lines).strip()
    return (text or None), {"ocr_provider": "paddleocr", "ocr_status": "ok" if text else "empty"}
