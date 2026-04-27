#!/usr/bin/env python3
"""Reproduce Sewer-ML random sampling with seed 42.

This script creates:
- single-label set: 50 images per defect class (exactly one defect code is 1)
- multi-label set: 50 images total (two or more defect codes are 1)

Default output naming:
- single label: <DEFECT_CODE>_<ORIGINAL_STEM>.png
- multi label:  <ORIGINAL_STEM>_<CODE1_CODE2_...>.png
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image


DEFECT_CODES: Sequence[str] = (
    "RB",
    "OB",
    "PF",
    "DE",
    "FS",
    "IS",
    "RO",
    "IN",
    "AF",
    "BE",
    "FO",
    "GR",
    "PH",
    "PB",
    "OS",
    "OP",
    "OK",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sewer-ML random sampler (seed 42).")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("/home/tai/workspace/ssd1/data/Sewer/Sewer-ML/SewerML_Train.csv"),
        help="Path to SewerML_Train.csv",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("/home/tai/workspace/ssd1/data/Sewer/Sewer-ML/train"),
        help="Directory containing original train images",
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=Path("/home/tai/workspace/choi/ICAM/data/single_label"),
        help="Output directory for single-label samples",
    )
    parser.add_argument(
        "--multi-output",
        type=Path,
        default=Path("/home/tai/workspace/choi/ICAM/data/multi_label"),
        help="Output directory for multi-label samples",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--per-class",
        type=int,
        default=50,
        help="Number of single-label samples per defect class",
    )
    parser.add_argument(
        "--multi-count",
        type=int,
        default=50,
        help="Number of multi-label samples in total",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing PNG files in output directories before writing",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            defects: List[str] = []
            for code in DEFECT_CODES:
                value = row.get(code, "0")
                try:
                    is_one = int(float(value)) == 1
                except (TypeError, ValueError):
                    is_one = str(value).strip() == "1"
                if is_one:
                    defects.append(code)

            row["_defects"] = defects
            rows.append(row)
    return rows


def build_filename_index(train_dir: Path) -> tuple[Dict[str, Path], Dict[str, List[Path]]]:
    exact_map: Dict[str, Path] = {}
    stem_map: Dict[str, List[Path]] = {}
    for path in sorted(train_dir.iterdir()):
        if not path.is_file():
            continue
        exact_map[path.name] = path
        stem_map.setdefault(path.stem, []).append(path)
    return exact_map, stem_map


def resolve_source(
    filename: str,
    exact_map: Dict[str, Path],
    stem_map: Dict[str, List[Path]],
) -> Optional[Path]:
    if filename in exact_map:
        return exact_map[filename]

    stem = Path(filename).stem
    candidates = stem_map.get(stem)
    if not candidates:
        return None

    suffix = Path(filename).suffix.lower()
    if suffix:
        for candidate in candidates:
            if candidate.suffix.lower() == suffix:
                return candidate

    return candidates[0]


def pick_samples(items: Iterable[dict], n: int, seed: int) -> List[dict]:
    sorted_items = sorted(items, key=lambda row: row.get("Filename", ""))
    k = min(n, len(sorted_items))
    return random.Random(seed).sample(sorted_items, k)


def ensure_output_dir(path: Path, clean_output: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not clean_output:
        return
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() == ".png":
            file.unlink()


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_dup{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def save_png(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        image.save(dst, format="PNG")


def main() -> int:
    args = parse_args()

    ensure_output_dir(args.single_output, args.clean_output)
    ensure_output_dir(args.multi_output, args.clean_output)

    rows = load_rows(args.csv_path)
    exact_map, stem_map = build_filename_index(args.train_dir)

    single_by_code: Dict[str, List[dict]] = {code: [] for code in DEFECT_CODES}
    multi_rows: List[dict] = []
    for row in rows:
        defects = row["_defects"]
        if len(defects) == 1:
            single_by_code[defects[0]].append(row)
        elif len(defects) >= 2:
            multi_rows.append(row)

    missing_sources: List[str] = []
    failed_saves: List[tuple[str, str]] = []
    written_single: Dict[str, int] = {code: 0 for code in DEFECT_CODES}
    written_multi = 0

    for code in DEFECT_CODES:
        samples = pick_samples(single_by_code[code], args.per_class, args.seed)
        for row in samples:
            filename = str(row.get("Filename", "")).strip()
            src = resolve_source(filename, exact_map, stem_map)
            if src is None:
                missing_sources.append(filename)
                continue

            output_name = f"{code}_{Path(filename).stem}.png"
            dst = unique_destination(args.single_output / output_name)
            try:
                save_png(src, dst)
                written_single[code] += 1
            except Exception as error:  # noqa: BLE001
                failed_saves.append((filename, str(error)))

    multi_samples = pick_samples(multi_rows, args.multi_count, args.seed)
    for row in multi_samples:
        filename = str(row.get("Filename", "")).strip()
        src = resolve_source(filename, exact_map, stem_map)
        if src is None:
            missing_sources.append(filename)
            continue

        defects: List[str] = row["_defects"]
        suffix = "_".join(defects) if defects else "NONE"
        output_name = f"{Path(filename).stem}_{suffix}.png"
        dst = unique_destination(args.multi_output / output_name)
        try:
            save_png(src, dst)
            written_multi += 1
        except Exception as error:  # noqa: BLE001
            failed_saves.append((filename, str(error)))

    print("=== Sampling Result ===")
    for code in DEFECT_CODES:
        print(f"{code}: source_single={len(single_by_code[code])}, written={written_single[code]}")
    print(f"multi(>=2 defects): source={len(multi_rows)}, written={written_multi}")
    print(f"missing_sources={len(missing_sources)}")
    print(f"failed_conversions={len(failed_saves)}")
    if missing_sources:
        print("missing_examples=", missing_sources[:10])
    if failed_saves:
        print("failed_examples=", failed_saves[:5])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
