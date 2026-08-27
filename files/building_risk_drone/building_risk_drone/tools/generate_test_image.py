"""
실물 카메라/드론 없이도 파이프라인을 검증할 수 있도록
균열이 있는 콘크리트 벽면 이미지를 합성해서 만드는 스크립트입니다.
데모 준비/개발 단계에서 활용하세요.

사용 예:
  python tools/generate_test_image.py --pattern diagonal --out sample_images/wall_diagonal_crack.jpg
  python tools/generate_test_image.py --pattern vertical  --out sample_images/wall_vertical_crack.jpg
  python tools/generate_test_image.py --pattern x         --out sample_images/wall_x_crack.jpg
"""
import argparse
import random
import cv2
import numpy as np


def make_wall(width=640, height=480, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(150, 180, size=(height, width), dtype=np.uint16)
    noise = rng.normal(0, 8, size=(height, width))
    wall = np.clip(base + noise, 0, 255).astype(np.uint8)
    wall = cv2.GaussianBlur(wall, (3, 3), 0)
    return cv2.cvtColor(wall, cv2.COLOR_GRAY2BGR)


def draw_crack(img, start, end, width=2, jitter=6, segments=25, color=(40, 40, 40)):
    x0, y0 = start
    x1, y1 = end
    pts = [(x0, y0)]
    for i in range(1, segments):
        t = i / segments
        x = x0 + (x1 - x0) * t + random.uniform(-jitter, jitter)
        y = y0 + (y1 - y0) * t + random.uniform(-jitter, jitter)
        pts.append((int(x), int(y)))
    pts.append((x1, y1))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, width, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_images/wall_diagonal_crack.jpg")
    ap.add_argument("--pattern", choices=["vertical", "horizontal", "diagonal", "x", "none"],
                     default="diagonal")
    args = ap.parse_args()

    img = make_wall()
    h, w = img.shape[:2]

    if args.pattern == "vertical":
        draw_crack(img, (w // 2, 20), (w // 2 + 15, h - 20), width=3)
    elif args.pattern == "horizontal":
        draw_crack(img, (20, h // 2), (w - 20, h // 2 + 10), width=3)
    elif args.pattern == "diagonal":
        draw_crack(img, (w // 4, h - 20), (3 * w // 4, 20), width=4)
    elif args.pattern == "x":
        draw_crack(img, (w // 4, h - 20), (3 * w // 4, 20), width=4)
        draw_crack(img, (w // 4, 20), (3 * w // 4, h - 20), width=4)
    # "none"이면 균열 없는 정상 벽면 (오탐 테스트용)

    cv2.imwrite(args.out, img)
    print(f"저장됨: {args.out}")


if __name__ == "__main__":
    main()
