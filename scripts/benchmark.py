from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.pipeline import DocumentPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OCR/document extraction.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json", action="store_true", dest="show_json")
    args = parser.parse_args()

    data = args.file.read_bytes()
    pipeline = DocumentPipeline(get_settings())

    print("Warming models...")
    pipeline.warmup()

    durations: list[float] = []
    last_result = None
    for index in range(args.runs):
        result = pipeline.extract_bytes(data, args.file.name)
        durations.append(float(result["total_seconds"]))
        last_result = result
        print(f"run {index + 1}: {durations[-1]:.3f}s")

    print()
    print(f"mean:   {statistics.mean(durations):.3f}s")
    print(f"median: {statistics.median(durations):.3f}s")
    if len(durations) > 1:
        print(f"min:    {min(durations):.3f}s")
        print(f"max:    {max(durations):.3f}s")

    if last_result:
        for page in last_result["pages"]:
            validation = page["validation"]
            print(
                f"page {page['page']}: OCR={page['ocr_seconds']:.3f}s "
                f"total={page['duration_seconds']:.3f}s "
                f"validation={validation['score']:.1f}%/{validation['status']} "
                f"tables={len(page['tables'])}"
            )

    if args.show_json and last_result:
        print(json.dumps(last_result, indent=2))


if __name__ == "__main__":
    main()
