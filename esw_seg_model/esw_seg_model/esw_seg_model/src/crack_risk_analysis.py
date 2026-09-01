"""
crack_risk_analysis.py
-----------------------------------------------------
단일 프레임의 crack segmentation 이진 마스크로부터
"물리적으로 의미 있는" 균열 지표(길이/폭/면적)와
위험도 점수를 계산하는 모듈.

핵심 아이디어
=============
- mask에서 균열 하나하나를 connected component로 분리
- 각 component에 대해 skeleton(중심선) + distance transform으로
  "폭"을 실제로 측정 (그냥 bounding box가 아니라, 균열처럼 가늘고 긴
  형태에 맞는 방식)
- 폭을 mm로 환산하려면 카메라 보정값이 필요함 (아래 주의 참고)
- 위험도는 "균열 폭" 기준 등급표를 기본으로 하되, 폭 정보가 없으면
  픽셀 기준 상대 심각도로 대체

주의 (반드시 읽어주세요)
=========================
- 모노큘러(단일) 웹캠에는 절대 스케일(mm) 정보가 없습니다. 실제 mm 단위
  폭을 구하려면 다음 중 하나가 필요합니다.
    1) mm_per_pixel을 직접 보정해서 넣기
       예: 화면에 크기를 아는 물체(예: 가로 50mm 카드)를 대고, 그 물체가
       화면에서 몇 픽셀로 찍히는지 재서 50 / pixel_width 로 계산
    2) 카메라-벽면 거리 + 초점거리 + 센서 크기를 이용한 핀홀 카메라 모델 보정
  보정값이 없으면 '픽셀 기준 상대 심각도'만 표시되며, 이는 실제 mm 폭과
  다를 수 있으므로 참고용입니다.
- 아래 균열 등급표(WIDTH_GRADE_TABLE_MM)는 흔히 쓰이는 콘크리트 균열폭
  등급 구간을 참고해 구성한 예시입니다. 실제 프로젝트에는 해당 구조물에
  적용되는 공식 기준(예: 국토안전관리원 정밀안전점검 지침, 관련 KCI/ACI
  기준 등)으로 반드시 교체해서 사용하세요.
- 이 스크립트가 산출하는 위험도는 자동화된 스크리닝 보조 지표입니다.
  실제 구조 안전 판정은 반드시 자격을 갖춘 전문가의 정밀 점검으로
  확인해야 합니다.
"""

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

MIN_COMPONENT_AREA_PX = 25  # 노이즈성 작은 blob 제거 임계값 (기존 15 -> 25로 상향)

# 화면(프레임) 면적 대비 이 비율보다 큰 단일 blob은 실제 균열이라기보다
# 조명/그림자/모델 오탐일 가능성이 높다고 보고 위험도 계산에서 제외합니다.
# (실제 균열이 화면의 15% 이상을 차지하는 경우는 극히 드묾)
# 더 엄격하게 걸러내고 싶으면 이 값을 낮추세요 (예: 0.08).
MAX_PLAUSIBLE_COMPONENT_AREA_RATIO = 0.15

# (미만 임계값 mm, 등급, 점수) - 마지막 구간(2.0mm 이상)은 WIDTH_GRADE_MAX 사용
# 실제 프로젝트에는 해당 구조물에 적용되는 공식 균열폭 등급 기준으로 교체하세요.
WIDTH_GRADE_TABLE_MM = [
    (0.1, "A", 5),
    (0.3, "B", 25),
    (1.0, "C", 50),
    (2.0, "D", 75),
]
WIDTH_GRADE_MAX = ("E", 100)


def width_mm_to_score(width_mm: float, sensitivity: float = 1.0):
    """sensitivity: 1.0이 기본. 0.5처럼 낮추면 같은 실제 폭이라도 등급을 더
    낮게(=더 빡빡하게) 매김. 1.5처럼 올리면 더 민감하게 반응."""
    effective_mm = width_mm * sensitivity
    for threshold, grade, score in WIDTH_GRADE_TABLE_MM:
        if effective_mm < threshold:
            return score, grade
    return WIDTH_GRADE_MAX[1], WIDTH_GRADE_MAX[0]


def _measure_component_geometry(comp_mask_u8: np.ndarray):
    """단일 crack connected-component에 대해
    (length_px, max_width_px, mean_width_px) 를 반환."""
    if _HAS_SKIMAGE:
        skel = skeletonize(comp_mask_u8 > 0)
        length_px = int(skel.sum())
        if length_px > 0:
            dist = cv2.distanceTransform(comp_mask_u8, cv2.DIST_L2, 5)
            widths = dist[skel] * 2.0  # distance transform: 중심선->배경까지 거리 = 반폭
            return length_px, float(widths.max()), float(widths.mean())

    # skimage 미설치거나 skeleton이 비어있는 아주 작은 blob인 경우의 fallback:
    # 최소 회전 사각형(minAreaRect)으로 길이/폭 근사
    contours, _ = cv2.findContours(comp_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0.0, 0.0
    c = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    length_px = max(rw, rh)
    width_px = min(rw, rh)
    return int(length_px), float(width_px), float(width_px)


def _relative_score_no_calibration(max_w_px, area_px, frame_shape):
    """mm 보정이 없을 때 픽셀 기준 상대 심각도(0~100). 절대 물리량이 아님에 주의."""
    h, w = frame_shape[:2]
    rel_width = max_w_px / float(h)       # 프레임 높이 대비 균열 폭 비율
    rel_area = area_px / float(h * w)     # 전체 면적 대비 균열 면적 비율
    score = min(100.0, (rel_width * 4000) + (rel_area * 3000))
    if score < 5:
        grade = "A"
    elif score < 25:
        grade = "B"
    elif score < 50:
        grade = "C"
    elif score < 75:
        grade = "D"
    else:
        grade = "E"
    return round(score, 1), grade


def analyze_mask(binary_mask_u8: np.ndarray, mm_per_pixel: float = None):
    """
    binary_mask_u8: 0/255 이진 마스크 (원본 프레임 해상도로 resize된 것 권장)
    mm_per_pixel: 보정값(선택). 있으면 mm 단위 필드가 채워지고 그것을 기준으로
                  등급을 매김. 없으면 픽셀 기준 상대 심각도를 사용.

    반환 dict:
        cracks: 개별 균열 리스트. 각 원소는
            bbox, centroid, area_px, length_px, max_width_px, mean_width_px,
            (mm_per_pixel 있으면) area_mm2, length_mm, max_width_mm, mean_width_mm,
            score(0~100), grade(A~E)
        num_cracks, total_crack_area_px, crack_area_ratio,
        frame_score, frame_grade  (해당 프레임에서 "가장 심각한 균열" 기준.
            평균이 아니라 최댓값을 쓰는 이유: 위험 판단은 평균으로 흐리면 안 되고
            가장 나쁜 지점을 기준으로 해야 안전측 판단이 됨)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask_u8, connectivity=8
    )

    cracks = []
    for i in range(1, num_labels):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < MIN_COMPONENT_AREA_PX:
            continue
        comp_mask = (labels == i).astype(np.uint8) * 255
        length_px, max_w_px, mean_w_px = _measure_component_geometry(comp_mask)

        crack = {
            "bbox": tuple(int(v) for v in stats[i, :4]),
            "centroid": tuple(float(v) for v in centroids[i]),
            "area_px": area_px,
            "length_px": length_px,
            "max_width_px": max_w_px,
            "mean_width_px": mean_w_px,
        }

        if mm_per_pixel:
            max_w_mm = max_w_px * mm_per_pixel
            crack.update({
                "area_mm2": area_px * (mm_per_pixel ** 2),
                "length_mm": length_px * mm_per_pixel,
                "max_width_mm": max_w_mm,
                "mean_width_mm": mean_w_px * mm_per_pixel,
            })
            score, grade = width_mm_to_score(max_w_mm)
        else:
            score, grade = _relative_score_no_calibration(
                max_w_px, area_px, binary_mask_u8.shape
            )

        crack["score"] = score
        crack["grade"] = grade
        cracks.append(crack)

    total_area_px = int((binary_mask_u8 > 0).sum())
    frame_h, frame_w = binary_mask_u8.shape[:2]
    crack_area_ratio = total_area_px / float(frame_h * frame_w)

    if cracks:
        frame_score = max(c["score"] for c in cracks)
        frame_grade = max(cracks, key=lambda c: c["score"])["grade"]
    else:
        frame_score, frame_grade = 0.0, "-"

    return {
        "cracks": cracks,
        "num_cracks": len(cracks),
        "total_crack_area_px": total_area_px,
        "crack_area_ratio": crack_area_ratio,
        "frame_score": frame_score,
        "frame_grade": frame_grade,
    }