"""
건물 위험도 판별 모듈 - 실행 진입점

사용 예:
  # 합성 테스트 이미지로 파이프라인 검증 (카메라 불필요)
  python main.py --source sample_images/wall_diagonal_crack.jpg

  # 라즈베리파이 CSI 카메라로 실시간 실행 (디스플레이 없이, SSH 원격)
  python main.py --source picam --headless

  # USB 웹캠으로 실시간 실행 (모니터 연결 시)
  python main.py --source 0
"""
import argparse
import time
import yaml
import cv2

from src.camera import FrameSource
from src.crack_detector import detect_crack_candidates
from src.feature_extractor import extract_features
from src.risk_engine import RiskEngine
from src.visualizer import draw_overlay
from src.alert import AlertLogger
from src.yolo_detector import YoloDamageDetector


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_frame(frame, cfg, yolo):
    """YOLO가 활성화돼 있으면 우선 사용하고, 아니면 classical CV로 자동 폴백."""
    features = None
    if yolo is not None and yolo.enabled:
        features = yolo.detect(frame)
    if not features:
        candidates = detect_crack_candidates(frame, cfg["detection"]["classical_cv"])
        features = [extract_features(c) for c in candidates]
    return features


def main():
    parser = argparse.ArgumentParser(description="건물 위험도 판별 모듈 (데모)")
    parser.add_argument("--source", default=None,
                         help="0=웹캠, picam=라즈베리파이 카메라, 또는 이미지/영상 파일 경로")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--headless", action="store_true",
                         help="디스플레이 없이 실행 (라즈베리파이 원격 SSH용)")
    parser.add_argument("--save-video", default=None, help="결과 영상을 저장할 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.source if args.source is not None else cfg["camera"]["source"]

    cam = FrameSource(source, cfg["camera"]["width"], cfg["camera"]["height"])
    engine = RiskEngine(cfg)
    alert = AlertLogger(cfg)
    yolo = YoloDamageDetector(cfg)

    writer = None
    frame_idx = 0
    frame_skip = max(cfg["camera"].get("frame_skip", 1), 1)
    last_features = []
    prev_t = time.time()
    fps = 0.0

    print("[INFO] 건물 위험도 판별 모듈 시작 (headless가 아니면 'q'로 종료)")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[INFO] 프레임을 더 이상 읽을 수 없습니다. 종료합니다.")
                break

            if frame_idx % frame_skip == 0:
                last_features = analyze_frame(frame, cfg, yolo)
            frame_idx += 1

            raw_score, _ = engine.compute_frame_score(last_features, frame.shape)

            if cam.is_static_image():
                # 정지 이미지는 실시간 스트림처럼 EMA가 워밍업될 시간이 없으므로,
                # 같은 프레임을 여러 번 통과시켜 즉시 정상상태(steady state) 점수로 안정화한다.
                for _ in range(cfg["temporal_smoothing"]["history_len"]):
                    smoothed_score, level = engine.update(raw_score)
            else:
                smoothed_score, level = engine.update(raw_score)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            vis = draw_overlay(frame.copy(), last_features, smoothed_score, level, fps)
            alert.maybe_alert(frame, smoothed_score, level)

            if args.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save_video, fourcc, 15, (w, h))
                writer.write(vis)

            if not args.headless:
                cv2.imshow("Building Risk Assessment", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(f"[frame {frame_idx}] score={smoothed_score:.1f} level={level} fps={fps:.1f} "
                      f"defects={len(last_features)}")

            if cam.is_static_image():
                if not args.headless:
                    cv2.waitKey(0)
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
