import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFECT_CODES = [
    "VA",
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
]

DEFAULT_CSV = Path("/home/tai/workspace/ssd1/data/Sewer/Sewer-ML/SewerML_Val.csv")
DEFAULT_IMAGE_ROOT = Path("/home/tai/workspace/ssd1/data/Sewer/Sewer-ML/valid")


def is_one(value):
    try:
        return int(float(str(value).strip())) == 1
    except (TypeError, ValueError):
        return False


def collect_mono_defect_images(csv_path: Path, image_root: Path):
    samples = []
    missing_count = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"Filename", "Defect", *DEFECT_CODES}
        missing_cols = sorted(required - fields)
        if missing_cols:
            raise RuntimeError(f"CSV is missing required columns: {', '.join(missing_cols)}")

        for row in reader:
            if not is_one(row.get("Defect")):
                continue

            active_codes = [code for code in DEFECT_CODES if is_one(row.get(code))]
            if len(active_codes) != 1:
                continue

            filename = str(row.get("Filename", "")).strip()
            if not filename:
                continue

            image_path = image_root / filename
            if image_path.exists():
                samples.append((image_path, active_codes[0]))
            else:
                missing_count += 1

    return samples, missing_count


def build_grid(samples, grid_size=5, cell_size=256, title_height=28, seed=None):
    rng = random.Random(seed)
    max_count = grid_size * grid_size
    picked = rng.sample(samples, k=min(max_count, len(samples)))

    cell_total_height = cell_size + title_height
    canvas = Image.new(
        "RGB",
        (grid_size * cell_size, grid_size * cell_total_height),
        (20, 20, 20),
    )
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (img_path, defect_code) in enumerate(picked):
        x = (idx % grid_size) * cell_size
        y = (idx // grid_size) * cell_total_height

        draw.rectangle([x, y, x + cell_size, y + title_height], fill=(35, 35, 35))
        label = f"{img_path.name} | {defect_code}"
        draw.text((x + 6, y + 7), label, fill=(240, 240, 240), font=font)

        with Image.open(img_path) as im:
            tile = ImageOps.fit(im.convert("RGB"), (cell_size, cell_size), method=resample)
            canvas.paste(tile, (x, y + title_height))

    return canvas, picked


def main():
    parser = argparse.ArgumentParser(
        description="Visualize random 5x5 images where Defect=1 and exactly one defect code is active."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to SewerML CSV")
    parser.add_argument("--root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Image directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mono_defect_grid_5x5.jpg"),
        help="Output image path",
    )
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size per side")
    parser.add_argument("--cell-size", type=int, default=256, help="Tile size in pixels")
    parser.add_argument("--title-height", type=int, default=28, help="Label area height in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--show", action="store_true", help="Open output using default image viewer")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if not args.root.exists():
        raise FileNotFoundError(f"Image directory not found: {args.root}")

    samples, missing_count = collect_mono_defect_images(args.csv, args.root)
    if not samples:
        raise RuntimeError("No images matched: Defect=1 and exactly one active defect code.")

    grid_img, picked = build_grid(
        samples,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        title_height=args.title_height,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(args.output)
    print(
        f"Saved: {args.output} (sampled {len(picked)} / candidates {len(samples)}, "
        f"missing files {missing_count})"
    )

    if args.show:
        grid_img.show()


if __name__ == "__main__":
    main()
