"""
학습 데이터 없이도 즉시 동작하는 균열/손상 후보 검출기.

파이프라인:
  그레이스케일 -> CLAHE(대비 강화) -> 가우시안 블러 -> Canny 엣지
  -> 형태학적 닫힘(끊긴 엣지 연결) -> 컨투어 추출 -> 길고 얇은 형태만 필터링

이 모듈은 "균열처럼 생긴 형태"를 기하학적으로 걸러내는 역할만 하고,
실제 위험도 판단(심각도 점수화)은 risk_engine.py 에서 수행합니다.
"""
import cv2
import numpy as np


def _auto_canny(image, sigma=0.33):
    """이미지 중간값 기반으로 Canny 임계값을 자동 산출 (조명 변화에 덜 민감하게)."""
    v = float(np.median(image))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def detect_crack_candidates(frame_bgr, cfg):
    """
    Returns:
        List[dict]: [{"contour": ndarray, "bbox": (x,y,w,h), "length_px": float}, ...]
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = _auto_canny(blurred)

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_len = cfg.get("min_crack_length", 40)
    min_ar = cfg.get("min_aspect_ratio", 4.0)
    branch_fill_th = cfg.get("branching_fill_ratio", 0.15)
    branch_min_len = min_len * 2.5

    candidates = []
    for c in contours:
        length = cv2.arcLength(c, closed=False)
        if length < min_len:
            continue

        # 축 정렬 boundingRect는 대각선(예: 45도) 형태에서 가로==세로가 되어
        # "길고 얇음" 판정을 못 하는 문제가 있어, 회전된 최소 사각형(minAreaRect)을 사용한다.
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        long_side = max(rw, rh)
        short_side = max(min(rw, rh), 1.0)
        aspect_ratio = long_side / short_side
        rect_area = max(rw * rh, 1.0)
        fill_ratio = cv2.contourArea(c) / rect_area

        is_thin_line = aspect_ratio >= min_ar
        # X자/망상형 균열은 두 선이 교차하면서 전체 윤곽의 종횡비가 1에 가까워져
        # 위 조건에서 걸러진다. 대신 "면적 대비 매우 성긴(fill_ratio가 낮은) 큰 구조물"이라는
        # 특징으로 별도 판정한다. (교차/망상 균열은 전단파괴 전조로 특히 위험도가 높다)
        is_branching_network = (not is_thin_line) and length >= branch_min_len and fill_ratio <= branch_fill_th

        if not (is_thin_line or is_branching_network):
            continue

        x, y, w, h = cv2.boundingRect(c)
        candidates.append({
            "contour": c,
            "bbox": (x, y, w, h),
            "length_px": float(length),
            "is_branching": is_branching_network,
        })

    return candidates
