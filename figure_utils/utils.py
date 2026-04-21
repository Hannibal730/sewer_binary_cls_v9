import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

clicked_points = []
preview_patch = None
ax = None
fig = None

BOX_HEIGHT = 80
MASK_COLOR_BGR = (170, 170, 170)


def make_mask_polygon(p1, p2, height):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-6:
        raise ValueError("두 점이 너무 가깝습니다.")

    u = v / length

    n1 = np.array([-u[1],  u[0]], dtype=np.float32)
    n2 = np.array([ u[1], -u[0]], dtype=np.float32)

    # 이미지 좌표계에서 아래쪽(+y) 방향 선택
    n = n1 if n1[1] > n2[1] else n2

    p3 = p2 + n * height
    p4 = p1 + n * height

    return np.array([p1, p2, p3, p4], dtype=np.int32)


def apply_polygon_mask(img, polygon, color_bgr):
    out = img.copy()
    cv2.fillConvexPoly(out, polygon, color_bgr)
    return out


def onclick(event):
    global clicked_points, preview_patch, ax, fig, BOX_HEIGHT, MASK_COLOR_BGR

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    if len(clicked_points) >= 2:
        return

    x, y = int(event.xdata), int(event.ydata)
    clicked_points.append((x, y))
    ax.plot(x, y, "ro")

    if len(clicked_points) == 2:
        poly = make_mask_polygon(clicked_points[0], clicked_points[1], BOX_HEIGHT)

        if preview_patch is not None:
            preview_patch.remove()

        face_color = tuple(c / 255.0 for c in MASK_COLOR_BGR[::-1])  # BGR -> RGB
        preview_patch = Polygon(
            poly,
            closed=True,
            facecolor=face_color,
            edgecolor="yellow",
            alpha=0.65
        )
        ax.add_patch(preview_patch)

        ax.plot(
            [clicked_points[0][0], clicked_points[1][0]],
            [clicked_points[0][1], clicked_points[1][1]],
            "b-"
        )

        fig.canvas.draw()


def main():
    global ax, fig, BOX_HEIGHT, MASK_COLOR_BGR, clicked_points, preview_patch

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="입력 PNG 경로")
    parser.add_argument("--output", required=True, help="출력 PNG 경로")
    parser.add_argument("--height", type=int, default=80, help="마스킹 박스 높이(px)")
    parser.add_argument("--color", nargs=3, type=int, default=[170, 170, 170], help="BGR 색상 예: 170 170 170")
    args = parser.parse_args()

    BOX_HEIGHT = args.height
    MASK_COLOR_BGR = tuple(args.color)

    image_path = Path(args.image)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"이미지 로드 실패: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    clicked_points = []
    preview_patch = None

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        "좌측 상단 클릭 -> 우측 상단 클릭\n"
        f"height = {BOX_HEIGHT}px\n"
        "창을 닫으면 저장"
    )
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(clicked_points) != 2:
        print("점 2개가 선택되지 않아 저장하지 않았습니다.")
        return

    poly = make_mask_polygon(clicked_points[0], clicked_points[1], BOX_HEIGHT)
    result = apply_polygon_mask(img_bgr, poly, MASK_COLOR_BGR)
    cv2.imwrite(str(output_path), result)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()