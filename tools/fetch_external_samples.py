from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen


SAMPLES = [
    {
        "name": "kalman_arxiv_1710.04055.pdf",
        "url": "https://arxiv.org/pdf/1710.04055",
        "description": "Kalman filtering PDF with formulas and figures.",
    },
    {
        "name": "calibre_demo.docx",
        "url": "https://calibre-ebook.com/downloads/demos/demo.docx",
        "description": "DOCX document with paragraphs, tables, and images.",
    },
    {
        "name": "financial_sample.xlsx",
        "url": "https://go.microsoft.com/fwlink/?LinkID=521962",
        "description": "Microsoft sample workbook with a wide data table.",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public sample documents for external smoke tests.")
    parser.add_argument("--out", default="data/external_samples", help="Output directory for downloaded files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        target = out_dir / sample["name"]
        if target.exists() and target.stat().st_size > 0 and not args.force:
            print(f"exists {target}")
            continue
        request = Request(sample["url"], headers={"User-Agent": "localMathRag-test/0.1"})
        print(f"download {sample['url']} -> {target}")
        with urlopen(request, timeout=90) as response:
            target.write_bytes(response.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
