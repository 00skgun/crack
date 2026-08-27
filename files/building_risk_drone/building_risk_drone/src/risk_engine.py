"""
==========================================================
 건물 위험도 판별의 핵심 로직 (프로젝트에서 가장 중요한 모듈)
==========================================================

전체 흐름:
  1) 프레임에서 검출된 손상 후보들(features)의 개별 심각도(severity)를 계산
     severity = w_length*길이정규화 + w_width*폭정규화 + w_orientation*방향가중치

  2) 프레임 전체 위험 점수(raw score, 0~100) = 아래 3가지의 가중합
     - 가장 심각한 손상 1건 (max)       : 국소적으로 매우 위험한 손상을 놓치지 않기 위함
     - 전체 평균 심각도 (mean)          : 전반적인 손상 정도
     - 손상 면적이 차지하는 비율 (coverage) : 손상이 넓게 퍼져 있는지

  3) 실시간 영상은 프레임마다 결과가 흔들릴 수 있으므로
     - 지수이동평균(EMA)으로 점수를 매끈하게 만들고
     - 등급 하락 시에는 더 보수적인 임계값(히스테리시스)을 적용해 깜빡임 방지

  4) 최종 점수를 4단계 등급(안전/주의/위험/심각)으로 매핑
"""
from collections import deque
import numpy as np

LEVEL_SAFE = "안전"
LEVEL_CAUTION = "주의"
LEVEL_DANGER = "위험"
LEVEL_CRITICAL = "심각"

_LEVEL_ORDER = [LEVEL_SAFE, LEVEL_CAUTION, LEVEL_DANGER, LEVEL_CRITICAL]


class RiskEngine:
    def __init__(self, cfg):
        rs = cfg["risk_scoring"]
        self.w_len = rs["weights"]["length"]
        self.w_wid = rs["weights"]["width"]
        self.w_ori = rs["weights"]["orientation"]
        self.orient_weight = rs["orientation_weight"]
        self.width_danger_px = rs["width_danger_px"]
        self.combo = rs["frame_score_combo"]

        ts = cfg["temporal_smoothing"]
        self.alpha = ts["ema_alpha"]
        self.history = deque(maxlen=ts["history_len"])
        self.hysteresis = ts["hysteresis"]

        lv = cfg["levels"]
        self.safe_max = lv["safe_max"]
        self.caution_max = lv["caution_max"]
        self.danger_max = lv["danger_max"]

        self._ema_score = 0.0
        self._current_level = LEVEL_SAFE

    # ---- 1) 개별 손상의 심각도 ----
    def _defect_severity(self, feat, frame_diag):
        length_norm = min(feat["length_px"] / (frame_diag * 0.5), 1.0)
        width_norm = min(feat["width_px"] / self.width_danger_px, 1.0)
        ori_w = self.orient_weight.get(feat["orientation"], 0.5)

        severity = (
            self.w_len * length_norm
            + self.w_wid * width_norm
            + self.w_ori * ori_w
        )
        return float(np.clip(severity, 0.0, 1.0)) * 100.0

    # ---- 2) 프레임 전체 점수 ----
    def compute_frame_score(self, features, frame_shape):
        h, w = frame_shape[:2]
        frame_diag = float(np.hypot(h, w))
        frame_area = float(h * w)

        if not features:
            return 0.0, []

        severities = [self._defect_severity(f, frame_diag) for f in features]
        coverage = sum(f["area_px"] for f in features) / frame_area
        coverage_score = min(coverage * 500.0, 100.0)  # 면적비율 -> 0~100 스케일 보정

        raw = (
            self.combo["max_w"] * max(severities)
            + self.combo["mean_w"] * (sum(severities) / len(severities))
            + self.combo["coverage_w"] * coverage_score
        )
        return float(np.clip(raw, 0, 100)), severities

    # ---- 3) 시간적 안정화 + 4) 등급 매핑 ----
    def update(self, raw_score):
        self._ema_score = self.alpha * raw_score + (1 - self.alpha) * self._ema_score
        self.history.append(self._ema_score)
        self._current_level = self._level_from_score(self._ema_score, self._current_level)
        return self._ema_score, self._current_level

    def _target_level(self, score):
        if score <= self.safe_max:
            return LEVEL_SAFE
        if score <= self.caution_max:
            return LEVEL_CAUTION
        if score <= self.danger_max:
            return LEVEL_DANGER
        return LEVEL_CRITICAL

    def _level_from_score(self, score, prev_level):
        target = self._target_level(score)

        # 위험도가 오르는 방향이면 즉시 반영 (안전이 최우선)
        if _LEVEL_ORDER.index(target) >= _LEVEL_ORDER.index(prev_level):
            return target

        # 내려가는 방향이면 margin 만큼 여유를 둬서 진동(깜빡임)을 방지
        down_margin = self.hysteresis.get("down_margin", 0)
        target_down = self._target_level(score + down_margin)
        if _LEVEL_ORDER.index(target_down) < _LEVEL_ORDER.index(prev_level):
            return target_down
        return prev_level
