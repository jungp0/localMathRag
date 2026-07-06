from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import imghdr
import re
import shutil
import struct

from .formula import stable_id


VisualType = str

MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
CAPTION_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:fig(?:ure)?|chart|plot|diagram|image|photo|table)\s*[\w.-]*"
    r"|(?:\u56fe|\u8868)\s*[\w.-]*"
    r")\s*[:.)\-\u3001]?\s+.+",
    re.I,
)

VISUAL_TYPE_KEYWORDS = {
    "formula_image": [
        "formula",
        "equation",
        "latex",
        "math",
        "\u516c\u5f0f",
        "\u65b9\u7a0b",
    ],
    "table_image": [
        "table",
        "spreadsheet",
        "\u8868\u683c",
    ],
    "screenshot": [
        "screenshot",
        "screen capture",
        "ui",
        "window",
        "\u622a\u56fe",
        "\u754c\u9762",
    ],
    "diagram": [
        "diagram",
        "flow",
        "architecture",
        "block",
        "state machine",
        "schematic",
        "\u6d41\u7a0b",
        "\u67b6\u6784",
        "\u6846\u56fe",
        "\u793a\u610f",
        "\u7ed3\u6784",
    ],
    "chart": [
        "chart",
        "plot",
        "graph",
        "curve",
        "histogram",
        "bar chart",
        "scatter",
        "trend",
        "\u56fe\u8868",
        "\u66f2\u7ebf",
        "\u67f1\u72b6",
        "\u6298\u7ebf",
        "\u6563\u70b9",
        "\u8d8b\u52bf",
    ],
    "photo": [
        "photo",
        "picture",
        "camera",
        "\u7167\u7247",
        "\u56fe\u7247",
    ],
}


@dataclass(slots=True)
class MarkdownVisual:
    alt: str
    src: str
    start: int
    end: int
    caption: str | None
    nearby_text: str
    visual_type: VisualType


def extract_markdown_visuals(text: str) -> list[MarkdownVisual]:
    visuals: list[MarkdownVisual] = []
    for match in MARKDOWN_IMAGE_PATTERN.finditer(text):
        alt = match.group("alt").strip()
        src = match.group("src").strip()
        nearby = nearby_text(text, match.start(), match.end())
        caption = alt or nearest_caption(nearby.splitlines()) or None
        visual_type = classify_visual_type(caption=caption, nearby_text=nearby, file_name=src)
        visuals.append(
            MarkdownVisual(
                alt=alt,
                src=src,
                start=match.start(),
                end=match.end(),
                caption=caption,
                nearby_text=nearby,
                visual_type=visual_type,
            )
        )
    return visuals


def classify_visual_type(
    *,
    caption: str | None = None,
    nearby_text: str | None = None,
    file_name: str | None = None,
    media_type: str | None = None,
    explicit_hint: str | None = None,
) -> VisualType:
    if explicit_hint:
        hint = explicit_hint.lower()
        if "chart" in hint or "plot" in hint:
            return "chart"
        if "diagram" in hint:
            return "diagram"
        if "table" in hint:
            return "table_image"
        if "screenshot" in hint:
            return "screenshot"
        if "formula" in hint or "equation" in hint:
            return "formula_image"

    caption_lower = (caption or "").lower()
    for visual_type in ["diagram", "chart", "screenshot", "table_image", "formula_image", "photo"]:
        if any(keyword.lower() in caption_lower for keyword in VISUAL_TYPE_KEYWORDS.get(visual_type, [])):
            return visual_type

    text = " ".join(part for part in [caption or "", nearby_text or "", file_name or "", media_type or ""] if part)
    lower = text.lower()
    for visual_type, keywords in VISUAL_TYPE_KEYWORDS.items():
        if any(keyword.lower() in lower for keyword in keywords):
            return visual_type
    return "unknown"


def is_caption_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4 or len(stripped) > 240:
        return False
    return bool(CAPTION_PATTERN.match(stripped))


def nearest_caption(lines: list[str], index: int | None = None, radius: int = 3) -> str | None:
    if not lines:
        return None
    if index is None:
        for line in lines:
            if is_caption_line(line):
                return line.strip()
        return None

    candidates: list[tuple[int, str]] = []
    for line_index, line in enumerate(lines):
        if is_caption_line(line):
            candidates.append((abs(line_index - index), line.strip()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates[0][0] <= radius else None


def nearby_text(text: str, start: int, end: int, window_chars: int = 900) -> str:
    left = max(0, start - window_chars // 2)
    right = min(len(text), end + window_chars // 2)
    return compact_text(text[left:right])


def build_visual_index_text(
    *,
    visual_type: str,
    caption: str | None,
    nearby_text_value: str | None,
    label: str | None = None,
    file_name: str | None = None,
) -> str:
    parts = [
        "visual_object",
        f"visual_type:{visual_type}",
        f"label:{label}" if label else "",
        f"file:{file_name}" if file_name else "",
        f"caption:{caption}" if caption else "",
        f"nearby_text:{nearby_text_value}" if nearby_text_value else "",
    ]
    return "\n".join(part for part in parts if part)


def compact_text(text: str | None, limit: int = 1200) -> str:
    if not text:
        return ""
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def save_visual_artifact(
    *,
    artifact_dir: Path,
    doc_id: str,
    payload: bytes,
    preferred_name: str,
    media_type: str | None = None,
) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = artifact_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    extension = extension_for_media(payload, media_type) or Path(preferred_name).suffix.lstrip(".") or "bin"
    stem = Path(preferred_name).stem or stable_id("visual", doc_id, preferred_name)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "visual"
    target = doc_dir / f"{safe_stem}.{extension}"
    counter = 2
    while target.exists():
        target = doc_dir / f"{safe_stem}_{counter}.{extension}"
        counter += 1
    target.write_bytes(payload)
    return str(target)


def copy_local_visual(
    *,
    source_doc: Path,
    image_ref: str,
    artifact_dir: Path,
    doc_id: str,
) -> str | None:
    if image_ref.startswith(("http://", "https://", "data:")):
        if image_ref.startswith("data:"):
            return save_data_uri(image_ref, artifact_dir, doc_id)
        return image_ref
    source = (source_doc.parent / image_ref).resolve()
    if not source.exists() or not source.is_file():
        return None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = artifact_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    target = doc_dir / source.name
    counter = 2
    while target.exists():
        target = doc_dir / f"{source.stem}_{counter}{source.suffix}"
        counter += 1
    shutil.copy2(source, target)
    return str(target)


def save_data_uri(uri: str, artifact_dir: Path, doc_id: str) -> str | None:
    try:
        header, encoded = uri.split(",", 1)
        media_type = header.split(";", 1)[0].removeprefix("data:")
        payload = base64.b64decode(encoded)
    except Exception:
        return None
    return save_visual_artifact(
        artifact_dir=artifact_dir,
        doc_id=doc_id,
        payload=payload,
        preferred_name="embedded_visual",
        media_type=media_type,
    )


def extension_for_media(payload: bytes, media_type: str | None = None) -> str | None:
    if media_type:
        mapping = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/gif": "gif",
            "image/bmp": "bmp",
            "image/tiff": "tiff",
            "image/svg+xml": "svg",
        }
        if media_type in mapping:
            return mapping[media_type]
    guessed = imghdr.what(None, payload)
    if guessed == "jpeg":
        return "jpg"
    return guessed


def image_size(payload: bytes) -> tuple[int | None, int | None]:
    if payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24:
        width, height = struct.unpack(">II", payload[16:24])
        return width, height
    if payload[:6] in {b"GIF87a", b"GIF89a"} and len(payload) >= 10:
        width, height = struct.unpack("<HH", payload[6:10])
        return width, height
    if payload.startswith(b"\xff\xd8"):
        return jpeg_size(payload)
    return None, None


def jpeg_size(payload: bytes) -> tuple[int | None, int | None]:
    index = 2
    while index + 9 < len(payload):
        if payload[index] != 0xFF:
            index += 1
            continue
        marker = payload[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(payload):
            break
        length = struct.unpack(">H", payload[index : index + 2])[0]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if index + 7 <= len(payload):
                height = struct.unpack(">H", payload[index + 3 : index + 5])[0]
                width = struct.unpack(">H", payload[index + 5 : index + 7])[0]
                return width, height
        index += max(length, 2)
    return None, None
