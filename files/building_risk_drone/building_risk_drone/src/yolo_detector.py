"""
선택적 YOLO 기반 손상 탐지기.

- ultralytics 패키지 + 커스텀 학습 가중치(.pt)가 있을 때만 활성화됩니다.
- 조건이 안 맞으면(config에서 off / 가중치 없음 / 패키지 미설치) enabled=False가 되고,
  상위 파이프라인(main.py)이 자동으로 classical CV 검출기로 폴백합니다.
- 라즈베리파이에서는 .pt(PyTorch) 그대로 쓰면 느리므로,
  학습 후 NCNN 또는 TFLite로 export해서 사용하는 것을 권장합니다. (README 참고)
"""
import os


class YoloDamageDetector:
    def __init__(self, cfg):
        self.enabled = False
        self.model = None
        self.conf_th = cfg["detection"].get("yolo_conf_threshold", 0.4)

        if not cfg["detection"].get("use_yolo", False):
            return

        model_path = cfg["detection"].get("yolo_model_path")
        if not model_path or not os.path.exists(model_path):
            print(f"[YOLO] 가중치 파일이 없어 classical CV 모드로 대체합니다: {model_path}")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.enabled = True
        except ImportError:
            print("[YOLO] ultralytics 미설치 (`pip install ultralytics`). classical CV 모드로 대체합니다.")

    def detect(self, frame_bgr):
        """
        classical CV 검출기와 동일한 스키마로 결과를 반환하여
        risk_engine이 검출 방식과 무관하게 동작하도록 합니다.
        """
        if not self.enabled:
            return None

        results = self.model.predict(frame_bgr, conf=self.conf_th, verbose=False)
        features = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                orientation = "vertical" if h >= w else "horizontal"
                features.append({
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "length_px": float(max(w, h)),
                    "width_px": float(max(min(w, h), 1)),
                    "angle_deg": 90.0 if h >= w else 0.0,
                    "orientation": orientation,
                    "area_px": float(w * h),
                    "cls": cls_name,
                    "conf": conf,
                })
        return features
