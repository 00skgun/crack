"""
위험도가 설정된 등급(trigger_level) 이상이 되면
스냅샷 이미지 저장 + JSON 로그 기록을 수행합니다.

실제 대회 시스템(드론 -> 지상국 통신)으로 확장하려면
maybe_alert() 안의 TODO 위치에 MQTT publish 또는 HTTP POST를 추가하면 됩니다.
"""
import os
import json
import time
import cv2

_LEVEL_ORDER = ["안전", "주의", "위험", "심각"]


class AlertLogger:
    def __init__(self, cfg):
        self.cfg = cfg["alert"]
        self.save_dir = self.cfg.get("save_dir", "alerts")
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, "risk_log.jsonl")
        self._last_alert_ts = 0.0

    def maybe_alert(self, frame, score, level):
        if not self.cfg.get("enabled", True):
            return

        trigger = self.cfg.get("trigger_level", "위험")
        if _LEVEL_ORDER.index(level) < _LEVEL_ORDER.index(trigger):
            return

        now = time.time()
        if now - self._last_alert_ts < self.cfg.get("cooldown_sec", 10):
            return
        self._last_alert_ts = now

        ts = time.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(self.save_dir, f"alert_{ts}.jpg")
        cv2.imwrite(img_path, frame)

        record = {"timestamp": ts, "score": round(score, 1), "level": level, "image": img_path}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # TODO: 실제 시스템에서는 여기서 지상국으로 MQTT/HTTP 전송
        print(f"[ALERT] {ts} level={level} score={score:.1f} saved={img_path}")
