"""
pi_webcam_crack_test.py (Raspberry Pi 지원 버젼)
---------------------------------------------------
학습된 crack segmentation 모델(seg_model.pth)을 라즈베리파이 카메라 영상에 실시간으로
적용하고 위험도를 계산하여 로깅하는 스크립트.

[라즈베리파이 변경점]
1. 카메라 하드웨어 캡처 해상도/프레임 제한(cap-width, cap-height, cap-fps) 추가
   - Pi Camera의 기본 고해상도 캡처로 인한 OOM(Out of Memory) 및 병목 현상 방지.

사용법 (Raspberry Pi):
    # 최신 라즈베리파이 OS(Bullseye/Bookworm) 환경에서는 libcamerify를 앞에 붙여야 할 수 있습니다.
    libcamerify python pi_webcam_crack_test.py --model seg_model.pth --camera 0 \
        --cap-width 640 --cap-height 480 --img-size 256 --zone "1F_로비"
"""

import argparse
import logging
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from crack_risk_analysis import analyze_mask
from building_risk_tracker import FrameTracker, BuildingRiskState


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("crack_risk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def build_model(num_classes: int = 1) -> torch.nn.Module:
    try:
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=None, weights_backbone=None
        )
    except TypeError:
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)

    model.classifier = DeepLabHead(2048, num_classes)
    return model


def load_model(weights_path: str, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    model = build_model(num_classes=1)
    state_dict = torch.load(weights_path, map_location=device)

    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        logger.info(f"Ignored unexpected keys (e.g. aux_classifier): {unexpected_keys}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model, frame_bgr, img_size, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))

    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    output = model(tensor)["out"][0, 0].detach().cpu().numpy()
    mask_norm = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mask_norm


def make_overlay(frame_bgr, binary_mask_u8):
    overlay = frame_bgr.copy()
    overlay[binary_mask_u8 > 0] = (0, 0, 255)
    blended = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)
    return blended


GRADE_COLORS = {
    "-": (200, 200, 200),
    "A": (0, 200, 0),
    "B": (0, 200, 200),
    "C": (0, 165, 255),
    "D": (0, 100, 255),
    "E": (0, 0, 255),
}


def draw_risk_box(img, frame_result, zone, building_state, fps):
    h, w = img.shape[:2]
    box_h = max(80, min(140, int(h * 0.22)))
    box_top = h - box_h

    overlay = img.copy()
    cv2.rectangle(overlay, (0, box_top), (w, h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.68, img, 0.32, 0, dst=img)

    frame_grade = frame_result["frame_grade"]
    frame_color = GRADE_COLORS.get(frame_grade, (255, 255, 255))

    z = building_state.data["zones"].get(zone)
    if z:
        flag = " *WIDENING*" if z["growth_flag"] else ""
        zone_text = f"Zone max: {z['max_score_ever']:.0f}({z['max_grade_ever']}){flag}"
        zone_color = GRADE_COLORS.get(z["max_grade_ever"], (255, 255, 255))
    else:
        zone_text, zone_color = "Zone max: -", (200, 200, 200)

    ov_grade = building_state.data["overall_grade"]
    ov_color = GRADE_COLORS.get(ov_grade, (255, 255, 255))

    line1 = (f"[{zone}]  Frame: {frame_result['frame_score']:.0f} ({frame_grade})  "
             f"cracks={frame_result['num_cracks']}   FPS:{fps:.1f}")
    line2 = f"{zone_text}    |    Bldg: {building_state.data['overall_score']:.0f} ({ov_grade})"

    font_scale1 = max(0.4, min(0.7, box_h / 150))
    font_scale2 = max(0.5, min(0.8, box_h / 120))

    y1 = box_top + int(box_h * 0.42)
    y2 = box_top + int(box_h * 0.82)

    cv2.putText(img, line1, (14, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale1, frame_color, 2, cv2.LINE_AA)
    cv2.putText(img, line2, (14, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale2, ov_color, 2, cv2.LINE_AA)

    cv2.rectangle(img, (0, box_top), (w - 1, h - 1), ov_color, thickness=3)
    return img


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Webcam crack segmentation + risk assessment")
    parser.add_argument("--model", type=str, default="esw_seg_model_best.pt", help="학습된 모델(.pt) 경로")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 장치 인덱스")
    
    # 라즈베리파이를 위한 하드웨어 캡처 설정 추가
    parser.add_argument("--cap-width", type=int, default=640, help="카메라 하드웨어 캡처 너비")
    parser.add_argument("--cap-height", type=int, default=480, help="카메라 하드웨어 캡처 높이")
    parser.add_argument("--cap-fps", type=int, default=15, help="카메라 하드웨어 캡처 프레임 제한")
    
    parser.add_argument("--img-size", type=int, default=256, help="모델 입력 크기")
    parser.add_argument("--threshold", type=int, default=127, help="초기 threshold 값 (0~255)")
    parser.add_argument("--display-width", type=int, default=640, help="표시 창의 너비(px)")
    parser.add_argument("--cpu", action="store_true", help="CPU 강제 사용 (라즈베리파이는 기본 CPU)")
    parser.add_argument("--zone", type=str, default="unspecified", help="현재 촬영 중인 구역/방 이름")
    parser.add_argument("--state-file", type=str, default="building_risk_state.json")
    parser.add_argument("--mm-per-pixel", type=float, default=None)
    parser.add_argument("--log-file", type=str, default="log.txt")
    parser.add_argument("--log-interval", type=float, default=5.0)
    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    # 라즈베리파이는 보통 cuda가 없으므로 자동으로 CPU가 선택됩니다.
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from: {args.model}")
    model = load_model(args.model, device, logger)
    logger.info("Model loaded.")

    # 카메라 초기화 및 하드웨어 해상도/프레임 설정 적용
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (index={args.camera}). libcamerify 명령어를 사용해보세요.")
    
    # [중요] 라즈베리파이에서 메모리/성능 병목을 막기 위해 캡처 단에서부터 해상도를 제한합니다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
    cap.set(cv2.CAP_PROP_FPS, args.cap_fps)

    # 실제 적용된 설정 확인 로직
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Camera initialized with resolution: {actual_w}x{actual_h}")

    window_name = "Pi Crack Segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", window_name, args.threshold, 255, lambda x: None)

    building_state = BuildingRiskState(args.state_file)
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    building_state.start_session(session_id)
    building_state.recompute_overall()

    logger.info(f"Session started | session_id={session_id} zone={args.zone}")

    frame_tracker = FrameTracker()
    current_zone = args.zone

    prev_time = time.time()
    fps = 0.0
    last_log_time = 0.0

    print("[INFO] 's': 캡처저장 / 'z': 구역변경 / 'b': 요약출력 / 'q': 종료")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("프레임을 읽지 못했습니다. 종료합니다.")
                break

            h, w = frame.shape[:2]

            mask_norm = predict_mask(model, frame, args.img_size, device)
            mask_resized = cv2.resize(mask_norm, (w, h), interpolation=cv2.INTER_LINEAR)

            thresh_val = cv2.getTrackbarPos("Threshold", window_name)
            _, binary_mask = cv2.threshold(mask_resized, thresh_val, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8)

            frame_result = analyze_mask(binary_mask, mm_per_pixel=args.mm_per_pixel)
            newly_confirmed = frame_tracker.update(frame_result["cracks"])
            for crack in newly_confirmed:
                building_state.record_confirmed_crack(current_zone, crack, session_id)
                logger.info(
                    f"Confirmed crack | zone={current_zone} track_id={crack['track_id']} "
                    f"score={crack['score']} grade={crack['grade']} "
                    f"width_px={crack['width_px']:.2f}"
                )
            if newly_confirmed:
                building_state.recompute_overall()

            overlay_bgr = make_overlay(frame, binary_mask)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            if now - last_log_time >= args.log_interval:
                z = building_state.data["zones"].get(current_zone)
                zone_max = z["max_score_ever"] if z else 0
                zone_grade = z["max_grade_ever"] if z else "-"
                logger.info(
                    f"Snapshot | zone={current_zone} frame_score={frame_result['frame_score']:.1f} "
                    f"FPS={fps:.1f} cracks={frame_result['num_cracks']} "
                    f"zone_max={zone_max}({zone_grade}) bldg={building_state.data['overall_grade']}"
                )
                last_log_time = now

            dw = args.display_width
            dh = int(h * dw / w)
            display_img = cv2.resize(overlay_bgr, (dw, dh))
            draw_risk_box(display_img, frame_result, current_zone, building_state, fps)

            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, display_img)
                logger.info(f"Saved capture: {fname}")
            elif key == ord("z"):
                new_zone = input("[INPUT] 새 구역(zone) 이름을 입력하세요: ").strip()
                if new_zone:
                    logger.info(f"Zone changed: {current_zone} -> {new_zone}")
                    current_zone = new_zone
            elif key == ord("b"):
                logger.info("Manual summary requested:\n" + building_state.summary_text())

    finally:
        cap.release()
        cv2.destroyAllWindows()
        building_state.end_session()
        logger.info("Session ended. Final building risk summary:\n" + building_state.summary_text())

if __name__ == "__main__":
    main()