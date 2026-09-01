"""
building_risk_tracker.py
-----------------------------------------------------
카메라로 건물을 돌아다니며(탐색하며) crack 탐지를 계속 수행할 때,
1) 같은 균열을 여러 프레임에서 중복 카운트하지 않도록 프레임간 추적(tracking)하고
2) 구역(zone)별 / 건물 전체 위험도를 세션(실행)을 넘어 이어서 누적·갱신하는 모듈.

설계 개요
=========
[프레임 단위 검출] -> [세션 내 균열 추적: FrameTracker] -> [영속 상태: BuildingRiskState]

1) FrameTracker (세션 내, 메모리에만 존재)
   - 매 프레임 검출된 균열들을 이전 프레임의 활성 track과 bbox IoU로 매칭
   - 동일 균열이 연속 CONFIRM_FRAMES 프레임 이상 검출되어야 '확정된 균열'로
     인정 -> 카메라 흔들림/조명 변화로 인한 단발성 오탐이 위험도에 그대로
     반영되는 것을 방지
   - 확정된 균열은 세션 동안 딱 한 번만 건물 상태에 반영됨. 카메라가 같은
     균열을 10초간 비추고 있어도 위험 점수가 계속 쌓이며 폭주하지 않음

2) BuildingRiskState (파일로 영속 저장 -> 재실행해도 이어서 누적됨)
   - 구역(zone)별로 "지금까지 관측된 가장 심각한 균열 상태"를 저장
   - 세션(실행)이 끝나면 세션 요약을 기록하고, 같은 구역의 직전 세션과
     비교해서 균열 폭(또는 점수)이 뚜렷하게 커졌으면 '진행성 균열 의심'
     플래그를 세우고 건물 위험도에 가산점을 줌
     -> 실제 구조안전 진단에서는 균열의 절대 크기 자체보다 "진행 여부
        (시간에 따라 커지는가)"가 훨씬 중요한 위험 신호이기 때문
   - overall_score = 0.6 * (가장 위험한 구역의 최댓값)
                   + 0.3 * (전체 구역 평균)
                   + (진행성 균열 발견 시 +15)
     건물 위험도를 평균만으로 산정하면 위험한 구역 하나가 다른 안전한
     구역들에 의해 희석되어 버리므로, 최댓값 비중을 높게 주고 평균으로
     "균열이 광범위하게 퍼져 있는지"도 일부 반영하는 구조

한계 (중요, 반드시 인지하고 사용)
=================================
- 별도의 마커나 실내 측위 없이는, 예를 들어 며칠 뒤 같은 곳을 다시
  촬영했을 때 "이게 정확히 이전에 본 그 균열인지"를 영상만으로 완벽히
  재식별할 수 없습니다. 그래서 이 모듈은 '개별 균열 ID'를 세션 너머로
  추적하지 않고, zone(구역) 단위로 "이 구역에서 관측된 최댓값의 추이"를
  추적합니다. 더 정밀하게(균열 1개 단위로) 추적하려면 고정 카메라 위치,
  QR/AR 마커, UWB 등 실내측위, 또는 SLAM 기반 3D 맵핑을 추가하는 것을
  권장합니다.
- 이 모듈의 위험도는 자동화된 스크리닝 보조 지표입니다. 실제 구조 안전
  판정은 반드시 자격을 갖춘 전문가의 정밀 점검으로 확인해야 합니다.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

CONFIRM_FRAMES = 3          # 몇 프레임 연속 검출되어야 '확정 균열'로 볼지 (환경/FPS에 맞게 조정)
IOU_MATCH_THRESHOLD = 0.3   # 프레임간 같은 균열로 볼 bbox IoU 최소값
TRACK_TIMEOUT_SEC = 2.0     # 이 시간 이상 안 보이면 track 종료(화면 이탈로 간주)
GROWTH_RATIO_THRESHOLD = 1.2  # 직전 세션 대비 20% 이상 커지면 '진행성 균열'로 판단


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _ActiveTrack:
    track_id: str
    bbox: tuple
    hits: int = 1
    confirmed: bool = False
    last_seen: float = field(default_factory=time.time)
    best_score: float = 0.0
    best_grade: str = "-"
    best_width_mm: Optional[float] = None
    best_width_px: float = 0.0


class FrameTracker:
    """세션(실행) 동안 프레임간 균열을 추적해서 '확정된 새 균열'만 골라낸다."""

    def __init__(self):
        self._tracks = {}  # track_id -> _ActiveTrack

    def update(self, cracks_this_frame):
        """
        cracks_this_frame: analyze_mask()가 반환한 crack dict 리스트
        반환: 이번 프레임에서 새롭게 '확정'된 crack 정보 리스트.
              (이미 확정된 균열이 계속 화면에 보이는 경우는 다시 반환하지
              않음 -> 위험도 중복 누적 방지)
        """
        now = time.time()
        newly_confirmed = []
        matched_ids = set()

        for crack in cracks_this_frame:
            best_id, best_iou = None, 0.0
            for tid, tr in self._tracks.items():
                if tid in matched_ids:
                    continue
                iou = _bbox_iou(tr.bbox, crack["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None and best_iou >= IOU_MATCH_THRESHOLD:
                tr = self._tracks[best_id]
                tr.bbox = crack["bbox"]
                tr.hits += 1
                tr.last_seen = now
                if crack["score"] > tr.best_score:
                    tr.best_score = crack["score"]
                    tr.best_grade = crack["grade"]
                    tr.best_width_mm = crack.get("max_width_mm")
                    tr.best_width_px = crack["max_width_px"]
                matched_ids.add(best_id)

                if not tr.confirmed and tr.hits >= CONFIRM_FRAMES:
                    tr.confirmed = True
                    newly_confirmed.append({
                        "track_id": tr.track_id,
                        "score": tr.best_score,
                        "grade": tr.best_grade,
                        "width_mm": tr.best_width_mm,
                        "width_px": tr.best_width_px,
                    })
            else:
                tid = uuid.uuid4().hex[:8]
                self._tracks[tid] = _ActiveTrack(
                    track_id=tid,
                    bbox=crack["bbox"],
                    best_score=crack["score"],
                    best_grade=crack["grade"],
                    best_width_mm=crack.get("max_width_mm"),
                    best_width_px=crack["max_width_px"],
                )

        stale = [tid for tid, tr in self._tracks.items() if now - tr.last_seen > TRACK_TIMEOUT_SEC]
        for tid in stale:
            del self._tracks[tid]

        return newly_confirmed

    def active_confirmed_count(self):
        return sum(1 for tr in self._tracks.values() if tr.confirmed)


def _score_to_grade(score):
    if score < 5:
        return "A"
    if score < 25:
        return "B"
    if score < 50:
        return "C"
    if score < 75:
        return "D"
    return "E"


class BuildingRiskState:
    """세션(실행)을 넘어 파일에 저장/로드되는 구역별·건물 전체 위험도 상태."""

    def __init__(self, state_path: str, building_id: str = "default"):
        self.state_path = state_path
        self.building_id = building_id
        self.data = self._load()
        self._session_buffer = {}
        self._session_id = None

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "building_id": self.building_id,
            "created_at": time.time(),
            "last_updated": time.time(),
            "zones": {},
            "history_log": [],
            "overall_score": 0,
            "overall_grade": "-",
        }

    def save(self):
        self.data["last_updated"] = time.time()
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.state_path)  # 원자적 저장 -> 중간에 꺼져도 파일이 깨지지 않음

    def _get_zone(self, zone: str):
        if zone not in self.data["zones"]:
            self.data["zones"][zone] = {
                "max_score_ever": 0,
                "max_grade_ever": "-",
                "max_width_mm_ever": None,
                "confirmed_crack_count_total": 0,
                "sessions": [],  # [{session_id, ts, num_confirmed, max_score, max_width_mm}]
                "growth_flag": False,
            }
        return self.data["zones"][zone]

    def start_session(self, session_id: str):
        self._session_buffer = {}
        self._session_id = session_id

    def record_confirmed_crack(self, zone: str, crack: dict, session_id: str):
        """세션 중 새로 '확정'된 균열 1건을 구역 통계에 반영."""
        z = self._get_zone(zone)
        if crack["score"] > z["max_score_ever"]:
            z["max_score_ever"] = crack["score"]
            z["max_grade_ever"] = crack["grade"]
            z["max_width_mm_ever"] = crack.get("width_mm")
        z["confirmed_crack_count_total"] += 1
        self._session_buffer.setdefault(zone, []).append(crack)

    def recompute_overall(self):
        """건물 전체 위험도를 즉시 재계산 (실시간 화면 표시용으로도 호출 가능)."""
        zones = self.data["zones"]
        if not zones:
            self.data["overall_score"] = 0
            self.data["overall_grade"] = "-"
            return

        scores = [z["max_score_ever"] for z in zones.values()]
        max_zone_score = max(scores)
        mean_zone_score = sum(scores) / len(scores)
        growth_bonus = 15 if any(z["growth_flag"] for z in zones.values()) else 0

        overall = 0.6 * max_zone_score + 0.3 * mean_zone_score + growth_bonus
        overall = min(100, overall)

        self.data["overall_score"] = round(overall, 1)
        self.data["overall_grade"] = _score_to_grade(overall)

    def end_session(self):
        """세션 종료 시 세션 요약 기록 + 직전 세션 대비 균열 진행(성장) 여부 판정."""
        for zone, cracks in self._session_buffer.items():
            z = self._get_zone(zone)
            max_score = max(c["score"] for c in cracks) if cracks else 0
            widths_mm = [c["width_mm"] for c in cracks if c.get("width_mm") is not None]
            max_width_mm = max(widths_mm) if widths_mm else None

            session_summary = {
                "session_id": self._session_id,
                "ts": time.time(),
                "num_confirmed": len(cracks),
                "max_score": max_score,
                "max_width_mm": max_width_mm,
            }
            z["sessions"].append(session_summary)
            z["sessions"] = z["sessions"][-20:]  # 최근 20세션만 보관

            z["growth_flag"] = False
            if len(z["sessions"]) >= 2:
                prev = z["sessions"][-2]
                if max_width_mm is not None and prev.get("max_width_mm"):
                    if max_width_mm > prev["max_width_mm"] * GROWTH_RATIO_THRESHOLD:
                        z["growth_flag"] = True
                elif prev.get("max_score") is not None:
                    # mm 보정이 없을 때는 점수 기준으로 대체 판단 (B등급 이상일 때만 유의미하게 취급)
                    if max_score >= 25 and max_score > prev["max_score"] * GROWTH_RATIO_THRESHOLD:
                        z["growth_flag"] = True

        self.recompute_overall()
        self.data["history_log"].append({
            "session_id": self._session_id,
            "ts": time.time(),
            "zones_touched": list(self._session_buffer.keys()),
        })
        self.data["history_log"] = self.data["history_log"][-100:]
        self.save()

    def summary_text(self):
        d = self.data
        lines = [f"Building overall risk: {d['overall_score']} ({d['overall_grade']})"]
        for zone, z in d["zones"].items():
            flag = "  [WIDENING - 진행성 균열 의심]" if z["growth_flag"] else ""
            lines.append(
                f"  - {zone}: max={z['max_score_ever']}({z['max_grade_ever']}) "
                f"confirmed_cracks={z['confirmed_crack_count_total']}{flag}"
            )
        return "\n".join(lines)
