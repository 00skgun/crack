"""
결과 오버레이 시각화.

주의: OpenCV의 cv2.putText는 한글(유니코드)을 지원하지 않아 물음표(?)로 깨집니다.
따라서 한글이 필요한 배너 텍스트는 PIL로 렌더링합니다.
시스템에 한글 폰트가 없는 경우 자동으로 영문 라벨로 대체되어(깨진 글자 대신)
데모가 계속 동작하도록 설계했습니다.

라즈베리파이에 한글 폰트가 없다면 아래처럼 설치하세요:
  sudo apt install -y fonts-noto-cjk
  (또는 나눔고딕 등 원하는 한글 ttf를 fonts/ 폴더에 넣고 _FONT_CANDIDATES에 경로 추가)
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LEVEL_COLORS = {
    "안전": (0, 200, 0),
    "주의": (0, 200, 255),
    "위험": (0, 140, 255),
    "심각": (0, 0, 255),
}

LEVEL_EN = {"안전": "SAFE", "주의": "CAUTION", "위험": "DANGER", "심각": "CRITICAL"}

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    os.path.join(os.path.dirname(__file__), "..", "fonts", "NanumGothic.ttf"),
]
_FONT_CACHE = {}


def _get_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    _FONT_CACHE[size] = font
    return font


def _put_text_kr(frame_bgr, text, org, font_size=26, color_bgr=(0, 0, 0)):
    font = _get_font(font_size)
    if font is None:
        # 한글 폰트가 없으면 물음표로 깨지는 대신 영문으로 안전하게 대체
        return frame_bgr, False

    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), True


def draw_overlay(frame, features, score, level, fps=None):
    color = LEVEL_COLORS.get(level, (255, 255, 255))

    for f in features:
        x, y, w, h = f["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{f.get('orientation', '?')} {f.get('width_px', 0):.1f}px"
        cv2.putText(frame, label, (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)

    banner_kr = f"위험도: {level}  (score={score:.1f}/100)"
    frame, ok = _put_text_kr(frame, banner_kr, (10, 5), font_size=22, color_bgr=(0, 0, 0))
    if not ok:
        banner_en = f"RISK: {LEVEL_EN.get(level, level)} (score={score:.1f}/100)"
        cv2.putText(frame, banner_en, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (max(frame.shape[1] - 120, 0), 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame
