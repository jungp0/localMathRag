from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any


MODEL_FILES = (
    "image_resizer.onnx",
    "encoder.onnx",
    "decoder.onnx",
    "tokenizer.json",
)
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def model_files_status(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir) if model_dir else Path()
    missing = [name for name in MODEL_FILES if not (root / name).is_file() or (root / name).stat().st_size <= 0]
    return {
        "model_dir": str(root) if model_dir else "",
        "model_ready": bool(model_dir and not missing),
        "missing_model_files": missing,
    }


def backend_status() -> dict[str, Any]:
    try:
        from rapid_latex_ocr import LaTeXOCR  # noqa: F401

        return {"backend_ready": True, "backend": "rapid_latex_ocr"}
    except Exception as exc:
        return {
            "backend_ready": False,
            "backend": "rapid_latex_ocr",
            "backend_error": str(exc),
        }


def _load_rapid_model(model_dir: str | Path):
    from rapid_latex_ocr import LaTeXOCR

    root = Path(model_dir).resolve()
    status = model_files_status(root)
    if not status["model_ready"]:
        missing = ", ".join(status["missing_model_files"])
        raise FileNotFoundError(f"formula OCR model files are missing: {missing}")

    cache_key = str(root)
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = LaTeXOCR(
                image_resizer_path=root / "image_resizer.onnx",
                encoder_path=root / "encoder.onnx",
                decoder_path=root / "decoder.onnx",
                tokenizer_json=root / "tokenizer.json",
            )
            _MODEL_CACHE[cache_key] = model
    return model


def recognize(image_path: str | Path, model_dir: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    model = _load_rapid_model(model_dir)
    with _INFERENCE_LOCK:
        result = model(str(image_path))
    latex = str(result[0] if isinstance(result, tuple) else result).strip()
    inference_seconds = float(result[1]) if isinstance(result, tuple) and len(result) > 1 else None
    return {
        "latex": latex,
        "text": latex,
        "runtime": {
            "backend": "rapid_latex_ocr",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "inference_seconds": round(inference_seconds, 3) if inference_seconds is not None else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LocalMathRAG formula OCR runner")
    parser.add_argument("--image")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    if args.health:
        print(json.dumps({**backend_status(), **model_files_status(args.model_dir)}))
        return 0

    if not args.image:
        print(json.dumps({"latex": "", "error": "--image is required"}))
        return 2

    image_path = Path(args.image)
    if not image_path.exists():
        print(json.dumps({"latex": "", "error": f"image not found: {image_path}"}))
        return 2

    try:
        print(json.dumps(recognize(image_path, args.model_dir)))
        return 0
    except Exception as exc:
        print(json.dumps({"latex": "", "text": "", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
