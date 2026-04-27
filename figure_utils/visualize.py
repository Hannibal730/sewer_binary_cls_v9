import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFAULT_CSV = Path("/home/tai/workspace/ssd1/data/Sewer/Sewer-ML/SewerML_Val.csv")
DEFAULT_IMAGE_ROOT = Path("/home/tai/workspace/ssd1/data/Sewer/Sewer-ML/valid")


def is_defect_one(value):
    try:
        return int(float(str(value).strip())) == 1
    except (TypeError, ValueError):
        return False


def collect_defect_images(csv_path: Path, image_root: Path):
    image_paths = []
    missing_count = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "Filename" not in (reader.fieldnames or []) or "Defect" not in (reader.fieldnames or []):
            raise RuntimeError("CSV에 'Filename' 또는 'Defect' 컬럼이 없습니다.")

        for row in reader:
            if not is_defect_one(row.get("Defect")):
                continue

            filename = str(row.get("Filename", "")).strip()
            if not filename:
                continue

            image_path = image_root / filename
            if image_path.exists():
                image_paths.append(image_path)
            else:
                missing_count += 1

    return image_paths, missing_count


def build_grid(images, grid_size=5, cell_size=256, title_height=26, seed=None):
    rng = random.Random(seed)
    max_count = grid_size * grid_size
    picked = rng.sample(images, k=min(max_count, len(images)))

    cell_total_height = cell_size + title_height
    canvas = Image.new(
        "RGB",
        (grid_size * cell_size, grid_size * cell_total_height),
        (20, 20, 20),
    )
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, img_path in enumerate(picked):
        x = (idx % grid_size) * cell_size
        y = (idx // grid_size) * cell_total_height

        # 파일명 영역 배경
        draw.rectangle([x, y, x + cell_size, y + title_height], fill=(35, 35, 35))
        draw.text((x + 6, y + 6), img_path.name, fill=(240, 240, 240), font=font)

        with Image.open(img_path) as im:
            tile = ImageOps.fit(im.convert("RGB"), (cell_size, cell_size), method=resample)
            canvas.paste(tile, (x, y + title_height))

    return canvas, picked


def main():
    parser = argparse.ArgumentParser(description="Defect=1 이미지 랜덤 5x5 그리드 생성")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="SewerML_Val.csv 경로")
    parser.add_argument("--root", type=Path, default=DEFAULT_IMAGE_ROOT, help="valid 이미지 폴더")
    parser.add_argument("--output", type=Path, default=Path("random_grid_5x5.jpg"), help="출력 이미지 경로")
    parser.add_argument("--grid-size", type=int, default=5, help="그리드 한 변의 칸 수")
    parser.add_argument("--cell-size", type=int, default=256, help="각 칸의 픽셀 크기")
    parser.add_argument("--title-height", type=int, default=26, help="파일명 영역 높이(px)")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--show", action="store_true", help="생성 후 OS 기본 이미지 뷰어로 열기")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV를 찾을 수 없습니다: {args.csv}")
    if not args.root.exists():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {args.root}")

    img_paths, missing_count = collect_defect_images(args.csv, args.root)
    if not img_paths:
        raise RuntimeError("Defect=1 조건을 만족하고 valid 폴더에 존재하는 이미지를 찾지 못했습니다.")

    grid_img, picked = build_grid(
        img_paths,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        title_height=args.title_height,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(args.output)
    print(
        f"저장 완료: {args.output} (샘플 {len(picked)}장, "
        f"Defect=1 후보 {len(img_paths)}장, CSV엔 있으나 파일 누락 {missing_count}장)"
    )

    if args.show:
        grid_img.show()


if __name__ == "__main__":
    main()
