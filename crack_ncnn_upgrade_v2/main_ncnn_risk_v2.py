"""
V2 integrated runner.

Changes from v1:
  - NCNN stays at 160x160
  - no full 640x480 segmentation-mask resize/processing
  - 160x160 mask is used for grouping/localization only
  - original 640x480 pixels are analyzed only inside crack ROIs
  - nearby fragments are merged and tiny noise is removed before counting cracks

Uses existing unchanged modules:
  src/camera.py
  src/alert.py
  src/risk_engine_mm_v1.py
  src/pump_controller_v1.py
  src/visualizer_mm_v1.py
"""
import argparse
import time
import yaml
import cv2

from src.camera import FrameSource
from src.alert import AlertLogger
from src.ncnn_seg_detector_v2 import NcnnCrackSegmenterV2
from src.crack_measurement_v2 import measure_cracks_from_small_mask
from src.risk_engine_mm_v1 import RiskEngineMM
from src.pump_controller_v1 import PumpController
from src.visualizer_mm_v1 import draw_overlay_mm


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="NCNN crack risk V2")
    parser.add_argument("--source", default=None,
                        help="0=USB webcam, picam=Raspberry Pi CSI, or image/video path")
    parser.add_argument("--config", default="config_ncnn_risk_v2.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-video", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.source if args.source is not None else cfg["camera"]["source"]

    cam = FrameSource(source, cfg["camera"]["width"], cfg["camera"]["height"])
    segmenter = NcnnCrackSegmenterV2(cfg)
    engine = RiskEngineMM(cfg)
    alert = AlertLogger(cfg)
    pump = PumpController(cfg)

    cal = cfg["calibration"]
    if not cal.get("ready", False):
        print("[WARN] calibration.ready=false")
        print("[WARN] mm values are PREVIEW ONLY; hardware spray is blocked.")

    writer = None
    frame_idx = 0
    frame_skip = max(int(cfg["camera"].get("frame_skip", 1)), 1)
    last_features = []
    last_infer_ms = 0.0
    last_measure_ms = 0.0
    last_analysis_ms = 0.0
    prev_t = time.time()
    display_fps = 0.0

    print("[INFO] NCNN crack risk V2 started. Press q to quit when display is enabled.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[INFO] No more frames. Exiting.")
                break

            analyzed_now = (frame_idx % frame_skip == 0)
            if analyzed_now:
                t_all = time.perf_counter()

                t0 = time.perf_counter()
                mask_small = segmenter.predict_mask_small(frame)
                last_infer_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                last_features = measure_cracks_from_small_mask(
                    frame_bgr=frame,
                    mask_small=mask_small,
                    mm_per_px=float(cal["mm_per_px"]),
                    cfg=cfg["measurement"],
                )
                last_measure_ms = (time.perf_counter() - t0) * 1000.0
                last_analysis_ms = (time.perf_counter() - t_all) * 1000.0

            frame_idx += 1

            raw_score, _, diagnostics = engine.compute_frame_score(last_features, frame.shape)
            if cam.is_static_image():
                for _ in range(int(cfg["temporal_smoothing"]["history_len"])):
                    smoothed_score, level = engine.update(raw_score)
            else:
                smoothed_score, level = engine.update(raw_score)

            max_width_mm = max((f.get("width_mm", 0.0) for f in last_features), default=0.0)
            should_spray = engine.should_spray(
                last_features,
                level,
                calibration_ready=bool(cal.get("ready", False)),
            )
            # Keep the v1 pump controller and all of its dry-run/cooldown logic.
            pump.update(should_spray, max_width_mm=max_width_mm, level=level)

            now = time.time()
            display_fps = 0.9 * display_fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            vis = draw_overlay_mm(
                frame.copy(),
                last_features,
                smoothed_score,
                level,
                fps=display_fps,
                infer_ms=last_infer_ms,
                calibration_ready=bool(cal.get("ready", False)),
            )

            # Extra timing line: unlike the old FPS display, this directly tells
            # you how long original-frame measurement costs.
            cv2.putText(
                vis,
                f"ROI measure={last_measure_ms:.0f}ms | total analysis={last_analysis_ms:.0f}ms",
                (8, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            alert.maybe_alert(frame, smoothed_score, level)

            if args.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save_video, fourcc, 15, (w, h))
                writer.write(vis)

            if not args.headless:
                cv2.imshow("NCNN Building Crack Risk V2", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif analyzed_now:
                merged_sources = sum(int(f.get("source_components", 1)) for f in last_features)
                print(
                    f"[analysis frame {frame_idx}] "
                    f"NCNN={last_infer_ms:.0f}ms ROI={last_measure_ms:.0f}ms total={last_analysis_ms:.0f}ms "
                    f"score={smoothed_score:.1f} level={level} "
                    f"groups={len(last_features)} source_components={merged_sources} "
                    f"maxW={max_width_mm:.3f}mm "
                    f"coverage={diagnostics['coverage_ratio']*100:.3f}% "
                    f"spray_request={should_spray}"
                )

            if cam.is_static_image():
                if not args.headless:
                    cv2.waitKey(0)
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pump.close()
        cam.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
