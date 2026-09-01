#!/usr/bin/env python3
"""Run NCNN crack detection and pulse a pump through a two-input motor driver."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from crack_risk_analysis import analyze_mask
from webcam_crack_ncnn import (
    DEFAULT_BIN,
    DEFAULT_PARAM,
    NcnnSegmenter,
    make_overlay,
    open_camera,
    parse_roi,
    predict_full_mask,
    setup_logger,
)


class PumpController:
    """Fail-safe wrapper for the IN3/IN4 pins of a motor driver."""

    def __init__(
        self,
        in3_pin: int,
        in4_pin: int,
        dry_run: bool,
        logger: logging.Logger,
    ) -> None:
        self.dry_run = dry_run
        self.logger = logger
        self._is_on = False
        self.in3 = None
        self.in4 = None

        if not dry_run:
            try:
                from gpiozero import DigitalOutputDevice
            except ImportError as exc:
                raise RuntimeError(
                    "gpiozero가 없습니다. "
                    "sudo apt install -y python3-gpiozero 후 다시 실행하세요."
                ) from exc

            # initial_value=False로 프로세스 시작 시 두 입력을 모두 LOW로 둔다.
            try:
                self.in3 = DigitalOutputDevice(in3_pin, initial_value=False)
                self.in4 = DigitalOutputDevice(in4_pin, initial_value=False)
            except Exception:
                # 두 번째 핀 초기화 중 실패해도 첫 번째 핀을 LOW로 되돌리고 해제한다.
                if self.in3 is not None:
                    self.in3.off()
                    self.in3.close()
                raise

        self.off(log_event=False)
        mode = "DRY-RUN" if dry_run else "GPIO"
        self.logger.info(f"Pump initialized | mode={mode} IN3=BCM{in3_pin} IN4=BCM{in4_pin}")

    @property
    def is_on(self) -> bool:
        return self._is_on

    def on(self) -> None:
        if self._is_on:
            return
        if not self.dry_run:
            self.in4.off()
            self.in3.on()
        self._is_on = True
        self.logger.warning("PUMP ON")

    def off(self, log_event: bool = True) -> None:
        was_on = self._is_on
        if not self.dry_run and self.in3 is not None and self.in4 is not None:
            self.in3.off()
            self.in4.off()
        self._is_on = False
        if log_event and was_on:
            self.logger.info("PUMP OFF")

    def close(self) -> None:
        self.off()
        if self.in3 is not None:
            self.in3.close()
        if self.in4 is not None:
            self.in4.close()


class CrackTrigger:
    """Debounce detections and require a clear scene before another trigger."""

    def __init__(self, confirm_frames: int, rearm_clear_frames: int) -> None:
        self.confirm_frames = confirm_frames
        self.rearm_clear_frames = rearm_clear_frames
        self.confirm_streak = 0
        self.clear_streak = 0
        self.armed = True

    def update(self, detected: bool, pump_is_on: bool) -> bool:
        if detected:
            self.confirm_streak += 1
            self.clear_streak = 0
        else:
            self.confirm_streak = 0
            self.clear_streak += 1
            if self.clear_streak >= self.rearm_clear_frames and not pump_is_on:
                self.armed = True

        should_trigger = (
            detected
            and self.armed
            and not pump_is_on
            and self.confirm_streak >= self.confirm_frames
        )
        if should_trigger:
            self.armed = False
        return should_trigger


def draw_pump_status(
    image: np.ndarray,
    crack_detected: bool,
    crack_ratio_percent: float,
    minimum_ratio_percent: float,
    confirm_streak: int,
    confirm_frames: int,
    pump: PumpController,
    armed: bool,
) -> None:
    height, width = image.shape[:2]
    panel_height = max(70, int(height * 0.2))
    panel_top = height - panel_height
    shade = image.copy()
    cv2.rectangle(shade, (0, panel_top), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.7, image, 0.3, 0, dst=image)

    detection_color = (0, 0, 255) if crack_detected else (0, 200, 0)
    pump_color = (0, 0, 255) if pump.is_on else (200, 200, 200)
    detection_text = (
        f"CRACK={'YES' if crack_detected else 'NO'} "
        f"area={crack_ratio_percent:.2f}% >= {minimum_ratio_percent:.2f}% "
        f"confirm={confirm_streak}/{confirm_frames}"
    )
    pump_text = f"PUMP={'ON' if pump.is_on else 'OFF'} {'ARMED' if armed else 'WAIT CLEAR'}"
    cv2.putText(
        image,
        detection_text,
        (10, panel_top + int(panel_height * 0.42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        detection_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        pump_text,
        (10, panel_top + int(panel_height * 0.82)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        pump_color,
        2,
        cv2.LINE_AA,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi CSI camera + NCNN crack detector + pump controller"
    )
    parser.add_argument("--param", type=Path, default=DEFAULT_PARAM)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cap-width", type=int, default=320)
    parser.add_argument("--cap-height", type=int, default=240)
    parser.add_argument("--cap-fps", type=int, default=5)
    parser.add_argument("--process-every", type=int, default=1)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--roi", type=parse_roi, default=None, help="x,y,width,height")
    parser.add_argument("--min-crack-area-percent", type=float, default=3.0)
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--rearm-clear-frames", type=int, default=3)
    parser.add_argument("--pump-seconds", type=float, default=2.0)
    parser.add_argument("--pump-in3-pin", type=int, default=17, help="BCM GPIO number")
    parser.add_argument("--pump-in4-pin", type=int, default=27, help="BCM GPIO number")
    parser.add_argument("--display-width", type=int, default=640)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="GPIO 출력 없이 감지 로직만 시험")
    parser.add_argument("--log-file", default="crack_pump.log")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.process_every < 1:
        parser.error("--process-every는 1 이상이어야 합니다")
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold는 0~255 범위여야 합니다")
    if not 0 <= args.min_crack_area_percent <= 100:
        parser.error("--min-crack-area-percent는 0~100 범위여야 합니다")
    if args.confirm_frames < 1 or args.rearm_clear_frames < 1:
        parser.error("확인/재대기 프레임 수는 1 이상이어야 합니다")
    if args.pump_seconds <= 0:
        parser.error("--pump-seconds는 0보다 커야 합니다")
    if args.pump_in3_pin == args.pump_in4_pin:
        parser.error("IN3와 IN4는 서로 다른 GPIO 핀이어야 합니다")

    cv2.setNumThreads(1)
    logger = setup_logger(args.log_file)
    pump = PumpController(
        args.pump_in3_pin,
        args.pump_in4_pin,
        args.dry_run,
        logger,
    )
    capture = None

    try:
        logger.info(
            f"Loading NCNN model | param={args.param} bin={args.bin} threads={args.threads}"
        )
        segmenter = NcnnSegmenter(
            args.param.resolve(),
            args.bin.resolve(),
            args.threads,
        )
        capture = open_camera(
            args.camera_backend,
            args.camera,
            args.cap_width,
            args.cap_height,
            args.cap_fps,
            logger,
        )
        logger.info(f"Camera: {capture.description}")

        window_name = "NCNN Crack Pump Controller"
        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_index = 0
        normalized_mask = None
        binary_mask = None
        crack_detected = False
        crack_ratio_percent = 0.0
        trigger = CrackTrigger(args.confirm_frames, args.rearm_clear_frames)
        pump_off_at = 0.0

        logger.info(
            "System started | "
            f"minimum_area={args.min_crack_area_percent:.2f}% "
            f"confirm_frames={args.confirm_frames} pump_seconds={args.pump_seconds:.2f}"
        )

        while True:
            now = time.monotonic()
            if pump.is_on and now >= pump_off_at:
                pump.off()

            success, frame = capture.read()
            if not success or frame is None:
                logger.error("카메라 프레임을 읽지 못했습니다")
                break

            should_process = frame_index % args.process_every == 0
            if should_process:
                normalized_mask = predict_full_mask(segmenter, frame, args.roi)
                _, binary_mask = cv2.threshold(
                    normalized_mask,
                    args.threshold,
                    255,
                    cv2.THRESH_BINARY,
                )
                result = analyze_mask(binary_mask)
                crack_ratio_percent = result["crack_area_ratio"] * 100.0
                crack_detected = (
                    result["num_cracks"] > 0
                    and crack_ratio_percent >= args.min_crack_area_percent
                )

                if trigger.update(crack_detected, pump.is_on):
                    logger.warning(
                        "Confirmed crack -> pump trigger | "
                        f"area={crack_ratio_percent:.2f}% "
                        f"threshold={args.min_crack_area_percent:.2f}%"
                    )
                    pump.on()
                    pump_off_at = time.monotonic() + args.pump_seconds

            frame_index += 1

            if not args.headless and binary_mask is not None:
                display = make_overlay(frame, binary_mask)
                display_height = int(display.shape[0] * args.display_width / display.shape[1])
                display = cv2.resize(display, (args.display_width, display_height))
                draw_pump_status(
                    display,
                    crack_detected,
                    crack_ratio_percent,
                    args.min_crack_area_percent,
                    trigger.confirm_streak,
                    args.confirm_frames,
                    pump,
                    trigger.armed,
                )
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Ctrl+C received")
    finally:
        pump.close()
        if capture is not None:
            capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
        logger.info("System ended | pump forced OFF")


if __name__ == "__main__":
    main()
