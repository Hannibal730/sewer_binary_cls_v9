# python utils.py --image "~.png" --output "~.png"

# 좌상단 클릭 -> 우하단 클릭 (박스 1개 생성)
# 다시 좌상단 클릭 -> 우하단 클릭 반복
# 필요하면 우클릭으로 마지막 박스 취소
# 창 닫으면 누적된 모든 박스가 한 번에 저장

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

clicked_points = []
mask_boxes = []
box_patches = []
preview_patch = None
first_point_artist = None
ax = None
fig = None

BLUR_SIGMA = 20.0


def make_mask_box(top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    if left == right or top == bottom:
        raise ValueError("박스 크기가 0입니다. 서로 다른 두 점을 선택하세요.")

    return left, top, right, bottom


def apply_blur_boxes(img, boxes, blur_sigma):
    out = img.copy()
    for left, top, right, bottom in boxes:
        roi = out[top:bottom, left:right]
        if roi.size == 0:
            continue
        blurred_roi = cv2.GaussianBlur(roi, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        out[top:bottom, left:right] = blurred_roi
    return out


def clear_active_selection():
    global clicked_points, preview_patch, first_point_artist, fig

    clicked_points = []

    if preview_patch is not None:
        preview_patch.remove()
        preview_patch = None

    if first_point_artist is not None:
        first_point_artist.remove()
        first_point_artist = None

    if fig is not None:
        fig.canvas.draw_idle()


def onmove(event):
    global clicked_points, preview_patch, ax, fig

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    if len(clicked_points) != 1:
        return

    x0, y0 = clicked_points[0]
    x1, y1 = int(event.xdata), int(event.ydata)
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)

    width = max(1, right - left)
    height = max(1, bottom - top)

    if preview_patch is None:
        preview_patch = Rectangle(
            (left, top),
            width,
            height,
            linewidth=1.5,
            edgecolor="yellow",
            linestyle="--",
            facecolor="none",
        )
        ax.add_patch(preview_patch)
    else:
        preview_patch.set_xy((left, top))
        preview_patch.set_width(width)
        preview_patch.set_height(height)

    fig.canvas.draw_idle()


def onclick(event):
    global clicked_points, mask_boxes, box_patches, preview_patch, first_point_artist
    global ax, fig

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    # 좌클릭: 점 선택 (2점마다 박스 1개 생성)
    if event.button == 1:
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append((x, y))

        if len(clicked_points) == 1:
            if first_point_artist is not None:
                first_point_artist.remove()
            first_point_artist, = ax.plot(x, y, "ro", markersize=5)
            fig.canvas.draw_idle()
            return

        if len(clicked_points) >= 2:
            try:
                box = make_mask_box(clicked_points[0], clicked_points[1])
            except ValueError as exc:
                print(str(exc))
                clear_active_selection()
                return

            mask_boxes.append(box)
            clicked_points = []
            if first_point_artist is not None:
                first_point_artist.remove()
                first_point_artist = None

            if preview_patch is not None:
                preview_patch.remove()
                preview_patch = None

            left, top, right, bottom = box
            patch = Rectangle(
                (left, top),
                right - left,
                bottom - top,
                facecolor="none",
                edgecolor="yellow",
                linewidth=1.5,
            )
            box_patches.append(patch)
            ax.add_patch(patch)
            fig.canvas.draw_idle()
        return

    # 우클릭: 마지막 박스 취소 (선택 중인 점이 있으면 점 선택 취소)
    if event.button == 3:
        if clicked_points:
            clear_active_selection()
            print("현재 선택 중인 점을 취소했습니다.")
            return

        if mask_boxes and box_patches:
            mask_boxes.pop()
            last_patch = box_patches.pop()
            last_patch.remove()
            print("마지막 박스를 삭제했습니다.")
            fig.canvas.draw_idle()


def main():
    global ax, fig, BLUR_SIGMA
    global clicked_points, mask_boxes, box_patches, preview_patch, first_point_artist

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="입력 PNG 경로")
    parser.add_argument("--output", required=True, help="출력 PNG 경로")
    parser.add_argument(
        "--height",
        type=int,
        default=80,
        help="하위 호환용 옵션(현재는 사용하지 않음)",
    )
    parser.add_argument("--blur-sigma", type=float, default=8.0, help="가우시안 블러 강도(기본값: 8.0)")
    args = parser.parse_args()

    if args.blur_sigma <= 0:
        raise ValueError("--blur-sigma는 0보다 커야 합니다.")

    BLUR_SIGMA = args.blur_sigma

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
    mask_boxes = []
    box_patches = []
    preview_patch = None
    first_point_artist = None

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        "좌상단 클릭 -> 우하단 클릭 (2클릭마다 박스 1개)\n"
        "첫 클릭 후 마우스 이동 시 박스 미리보기 표시\n"
        "저장 시 박스 영역 블러 처리, 우클릭으로 선택 취소/마지막 박스 삭제\n"
        "창을 닫으면 저장"
    )
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("motion_notify_event", onmove)
    plt.show()

    if not mask_boxes:
        print("선택된 박스가 없어 저장하지 않았습니다.")
        return

    result = apply_blur_boxes(img_bgr, mask_boxes, BLUR_SIGMA)
    cv2.imwrite(str(output_path), result)
    print(f"저장 완료: {output_path} (박스 {len(mask_boxes)}개, blur_sigma={BLUR_SIGMA})")


if __name__ == "__main__":
    main()
