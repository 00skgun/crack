#!/usr/bin/env python3
"""Raspberry Pi 4 optimized NCNN crack segmentation and risk monitoring."""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import ncnn
import numpy as np

from building_risk_tracker import BuildingRiskState, FrameTracker
from crack_risk_analysis import analyze_mask


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "esw_seg_model_ncnn"
DEFAULT_PARAM = DEFAULT_MODEL_DIR / "esw_seg_model_fp16_160.ncnn.param"
DEFAULT_BIN = DEFAULT_MODEL_DIR / "esw_seg_model_fp16_160.ncnn.bin"
MODEL_INPUT_SIZE = 160


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("crack_risk_ncnn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class OpenCVCamera:
    """USB/V4L2 camera adapter with the same interface as Picamera2Camera."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV 카메라를 열 수 없습니다: index={index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.description = (
            f"opencv index={index} "
            f"{self.capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
            f"{self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ "
            f"{self.capture.get(cv2.CAP_PROP_FPS):.1f} FPS"
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class Picamera2Camera:
    """Raspberry Pi CSI camera adapter using the supported libcamera stack."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2가 없습니다. "
                "sudo apt install -y python3-picamera2 후 "
                "--system-site-packages 가상환경을 사용하세요."
            ) from exc

        self.camera = Picamera2(index)
        configuration = self.camera.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)},
            controls={"FrameRate": float(fps)},
            buffer_count=2,
        )
        self.camera.configure(configuration)
        self.camera.start()
        # 자동 노출과 화이트 밸런스가 첫 프레임 전에 안정화될 시간을 준다.
        time.sleep(0.5)
        stream = self.camera.stream_configuration("main")
        actual_width, actual_height = stream["size"]
        self.description = (
            f"picamera2 index={index} {actual_width}x{actual_height} @ {fps} FPS"
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            # Picamera2의 RGB888 배열은 OpenCV가 바로 처리할 수 있는 3채널 배열이다.
            frame = self.camera.capture_array("main")
        except Exception:
            return False, None
        if frame is None or frame.size == 0:
            return False, None
        return True, np.ascontiguousarray(frame)

    def release(self) -> None:
        try:
            self.camera.stop()
        finally:
            self.camera.close()


def open_camera(
    backend: str,
    index: int,
    width: int,
    height: int,
    fps: int,
    logger: logging.Logger,
) -> OpenCVCamera | Picamera2Camera:
    if backend == "picamera2":
        return Picamera2Camera(index, width, height, fps)
    if backend == "opencv":
        return OpenCVCamera(index, width, height, fps)

    try:
        camera = Picamera2Camera(index, width, height, fps)
        logger.info("CSI 카메라를 Picamera2로 열었습니다")
        return camera
    except (RuntimeError, OSError) as picamera_error:
        logger.info(f"Picamera2를 사용할 수 없어 OpenCV 카메라로 전환: {picamera_error}")
        try:
            return OpenCVCamera(index, width, height, fps)
        except RuntimeError as opencv_error:
            raise RuntimeError(
                "카메라 초기화 실패. CSI 카메라는 rpicam-hello로 동작 여부를 확인하고 "
                "python3-picamera2 설치 여부를 확인하세요. "
                f"Picamera2: {picamera_error}; OpenCV: {opencv_error}"
            ) from opencv_error


class NcnnSegmenter:
    def __init__(self, param_path: Path, bin_path: Path, threads: int) -> None:
        if not param_path.is_file():
            raise FileNotFoundError(f"NCNN param 파일이 없습니다: {param_path}")
        if not bin_path.is_file():
            raise FileNotFoundError(f"NCNN bin 파일이 없습니다: {bin_path}")

        self.threads = threads
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.lightmode = True
        self.net.opt.use_packing_layout = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        # Raspberry Pi 4 Cortex-A72는 ARMv8.2 FP16 산술 명령이 없으므로 FP32 누산.
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_vulkan_compute = False

        if self.net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"NCNN param 로드 실패: {param_path}")
        if self.net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"NCNN bin 로드 실패: {bin_path}")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        input_mat = ncnn.Mat.from_pixels_resize(
            np.ascontiguousarray(frame_bgr),
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            width,
            height,
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE,
        )
        input_mat.substract_mean_normalize(
            [0.0, 0.0, 0.0],
            [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
        )

        for _ in range(3):
            extractor = self.net.create_extractor()
            extractor.set_light_mode(True)
            if hasattr(extractor, "set_num_threads"):
                extractor.set_num_threads(self.threads)
            if extractor.input("in0", input_mat) != 0:
                raise RuntimeError("NCNN 입력 설정 실패")
            result, output = extractor.extract("out0")
            if result != 0:
                raise RuntimeError(f"NCNN 추론 실패: code={result}")

            logits = np.asarray(output).squeeze()
            if logits.shape == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE) and np.isfinite(logits).all():
                return cv2.normalize(logits, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )
        raise RuntimeError("NCNN이 3회 연속 비정상 출력을 반환했습니다")


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        roi = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI 형식은 x,y,width,height입니다") from exc
    if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
        raise argparse.ArgumentTypeError("ROI 형식은 x,y,width,height이며 크기는 양수여야 합니다")
    return roi


def predict_full_mask(
    segmenter: NcnnSegmenter,
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    if roi is None:
        mask = segmenter.predict(frame)
        return cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

    x, y, width, height = roi
    x1 = max(0, min(x, frame_width - 1))
    y1 = max(0, min(y, frame_height - 1))
    x2 = max(x1 + 1, min(x + width, frame_width))
    y2 = max(y1 + 1, min(y + height, frame_height))
    crop = frame[y1:y2, x1:x2]
    crop_mask = segmenter.predict(crop)
    crop_mask = cv2.resize(
        crop_mask,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    )
    full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = crop_mask
    return full_mask


def make_overlay(frame_bgr: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    overlay = frame_bgr.copy()
    overlay[binary_mask > 0] = (0, 0, 255)
    return cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)


GRADE_COLORS = {
    "-": (200, 200, 200),
    "A": (0, 200, 0),
    "B": (0, 200, 200),
    "C": (0, 165, 255),
    "D": (0, 100, 255),
    "E": (0, 0, 255),
}


def draw_risk_box(
    image: np.ndarray,
    frame_result: dict,
    zone: str,
    building_state: BuildingRiskState,
    fps: float,
    inference_ms: float,
) -> None:
    height, width = image.shape[:2]
    box_height = max(80, min(140, int(height * 0.22)))
    box_top = height - box_height
    overlay = image.copy()
    cv2.rectangle(overlay, (0, box_top), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, dst=image)

    frame_grade = frame_result["frame_grade"]
    frame_color = GRADE_COLORS.get(frame_grade, (255, 255, 255))
    zone_state = building_state.data["zones"].get(zone)
    if zone_state:
        flag = " *WIDENING*" if zone_state["growth_flag"] else ""
        zone_text = (
            f"Zone max: {zone_state['max_score_ever']:.0f}"
            f"({zone_state['max_grade_ever']}){flag}"
        )
    else:
        zone_text = "Zone max: -"

    overall_grade = building_state.data["overall_grade"]
    overall_color = GRADE_COLORS.get(overall_grade, (255, 255, 255))
    line1 = (
        f"[{zone}] risk={frame_result['frame_score']:.0f}({frame_grade}) "
        f"cracks={frame_result['num_cracks']} FPS={fps:.1f} infer={inference_ms:.0f}ms"
    )
    line2 = (
        f"{zone_text} | Building: {building_state.data['overall_score']:.0f}"
        f"({overall_grade})"
    )
    scale1 = max(0.42, min(0.65, box_height / 165))
    scale2 = max(0.5, min(0.75, box_height / 140))
    cv2.putText(
        image,
        line1,
        (10, box_top + int(box_height * 0.42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale1,
        frame_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        line2,
        (10, box_top + int(box_height * 0.82)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale2,
        overall_color,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (0, box_top), (width - 1, height - 1), overall_color, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Raspberry Pi 4 NCNN crack monitor")
    parser.add_argument("--param", type=Path, default=DEFAULT_PARAM)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
        help="auto는 CSI Picamera2를 먼저 시도하고 실패하면 OpenCV를 사용합니다",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cap-width", type=int, default=320)
    parser.add_argument("--cap-height", type=int, default=240)
    parser.add_argument("--cap-fps", type=int, default=5)
    parser.add_argument("--process-every", type=int, default=3)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--display-width", type=int, default=640)
    parser.add_argument("--roi", type=parse_roi, default=None, help="x,y,width,height")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--zone", type=str, default="unspecified")
    parser.add_argument("--state-file", type=str, default="building_risk_state.json")
    parser.add_argument("--mm-per-pixel", type=float, default=None)
    parser.add_argument("--log-file", type=str, default="log_ncnn.txt")
    parser.add_argument("--log-interval", type=float, default=5.0)
    args = parser.parse_args()

    if args.process_every < 1:
        parser.error("--process-every는 1 이상이어야 합니다")
    if args.threads < 1:
        parser.error("--threads는 1 이상이어야 합니다")

    cv2.setNumThreads(1)
    logger = setup_logger(args.log_file)
    logger.info(
        f"Loading NCNN model | param={args.param} bin={args.bin} threads={args.threads}"
    )
    segmenter = NcnnSegmenter(args.param.resolve(), args.bin.resolve(), args.threads)

    capture = open_camera(
        args.camera_backend,
        args.camera,
        args.cap_width,
        args.cap_height,
        args.cap_fps,
        logger,
    )
    logger.info(
        f"Camera: {capture.description} | process_every={args.process_every}"
    )

    window_name = "NCNN Crack Risk Monitor"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Threshold", window_name, args.threshold, 255, lambda _: None)

    building_state = BuildingRiskState(args.state_file)
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    building_state.start_session(session_id)
    building_state.recompute_overall()
    frame_tracker = FrameTracker()
    current_zone = args.zone
    frame_index = 0
    normalized_mask = None
    frame_result = None
    inference_ms = 0.0
    fps = 0.0
    previous_time = time.time()
    last_log_time = 0.0

    logger.info(f"Session started | id={session_id} zone={current_zone}")
    if args.headless:
        logger.info("Headless mode: Ctrl+C로 종료")
    else:
        print("[INFO] 's': 저장 / 'z': 구역 변경 / 'b': 요약 / 'q': 종료")

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                logger.error(
                    "프레임을 읽지 못했습니다. CSI 카메라는 "
                    "--camera-backend picamera2로 실행하세요."
                )
                break

            should_process = frame_index % args.process_every == 0
            if should_process:
                started = time.perf_counter()
                normalized_mask = predict_full_mask(segmenter, frame, args.roi)
                inference_ms = (time.perf_counter() - started) * 1000.0

            threshold = (
                args.threshold
                if args.headless
                else cv2.getTrackbarPos("Threshold", window_name)
            )
            _, binary_mask = cv2.threshold(
                normalized_mask,
                threshold,
                255,
                cv2.THRESH_BINARY,
            )

            if should_process:
                frame_result = analyze_mask(binary_mask, mm_per_pixel=args.mm_per_pixel)
                newly_confirmed = frame_tracker.update(frame_result["cracks"])
                for crack in newly_confirmed:
                    building_state.record_confirmed_crack(current_zone, crack, session_id)
                    logger.info(
                        f"Confirmed crack | zone={current_zone} score={crack['score']} "
                        f"grade={crack['grade']} width_px={crack['width_px']:.2f}"
                    )
                if newly_confirmed:
                    building_state.recompute_overall()

            frame_index += 1
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - previous_time, 1e-6)
            previous_time = now

            if now - last_log_time >= args.log_interval:
                logger.info(
                    f"Snapshot | zone={current_zone} score={frame_result['frame_score']:.1f} "
                    f"grade={frame_result['frame_grade']} cracks={frame_result['num_cracks']} "
                    f"inference={inference_ms:.0f}ms loop_fps={fps:.1f}"
                )
                last_log_time = now

            key = -1
            if not args.headless:
                display = make_overlay(frame, binary_mask)
                display_height = int(display.shape[0] * args.display_width / display.shape[1])
                display = cv2.resize(display, (args.display_width, display_height))
                draw_risk_box(
                    display,
                    frame_result,
                    current_zone,
                    building_state,
                    fps,
                    inference_ms,
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, display)
                logger.info(f"Saved capture: {filename}")
            elif key == ord("z"):
                new_zone = input("[INPUT] 새 구역 이름: ").strip()
                if new_zone:
                    current_zone = new_zone
            elif key == ord("b"):
                logger.info("Manual summary:\n" + building_state.summary_text())
    except KeyboardInterrupt:
        logger.info("Ctrl+C received")
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
        building_state.end_session()
        logger.info("Session ended:\n" + building_state.summary_text())


if __name__ == "__main__":
    main()
