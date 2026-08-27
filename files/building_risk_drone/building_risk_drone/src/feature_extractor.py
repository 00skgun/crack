"""
검출된 균열 후보(컨투어) 하나하나로부터 위험도 판단에 쓸 특징을 계산합니다.

- length_px    : 균열 길이 (호 길이)
- width_px     : 균열 폭 추정치 (면적 / 길이)
- angle_deg    : 균열의 주된 방향 (0~180도, cv2.fitLine 기반 직선근사)
- orientation  : vertical / horizontal / diagonal 분류

방향 분류는 구조공학의 일반적인 경험칙을 단순화한 것입니다:
  - 수직 균열   : 대체로 건조수축 등 경미한 원인, 위험도 낮음~중간
  - 수평 균열   : 휨(bending) 관련, 중간~높음
  - 대각선 균열 : 전단(shear) 관련, 구조적으로 가장 위험한 패턴으로 알려짐
  (실제 구조 진단은 반드시 전문가 확인이 필요하며, 본 로직은 데모용 휴리스틱입니다.)
"""
import cv2
import numpy as np

ORIENT_VERTICAL = "vertical"
ORIENT_HORIZONTAL = "horizontal"
ORIENT_DIAGONAL = "diagonal"
ORIENT_BRANCHING = "branching"  # X자형/망상형 (교차 균열) - 전단파괴 전조로 가장 위험


def _classify_orientation(angle_deg):
    a = angle_deg % 180
    if 70 <= a <= 110:
        return ORIENT_VERTICAL
    if a <= 20 or a >= 160:
        return ORIENT_HORIZONTAL
    return ORIENT_DIAGONAL


def extract_features(candidate):
    contour = candidate["contour"]
    length = candidate["length_px"]
    area = cv2.contourArea(contour)
    width_est = max(area / length, 1.0) if length > 0 else 1.0

    pts = contour.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    angle_deg = float(np.degrees(np.arctan2(vy, vx))) % 180

    if candidate.get("is_branching"):
        # 교차/망상형 균열은 하나의 대표 각도로 의미를 부여하기 어려우므로 별도 분류
        orientation = ORIENT_BRANCHING
    else:
        orientation = _classify_orientation(angle_deg)

    x, y, w, h = candidate["bbox"]
    return {
        "bbox": (x, y, w, h),
        "length_px": length,
        "width_px": float(width_est),
        "angle_deg": angle_deg,
        "orientation": orientation,
        "area_px": float(area),
    }
