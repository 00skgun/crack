"""
webcam_crack_test.py (risk-aware version)
---------------------------------------------------
학습된 crack segmentation 모델(seg_model.pth)을 웹캠 영상에 실시간으로
적용하고, 그 결과로부터
  1) 프레임 단위 균열 위험도(폭/길이/면적 기반)
  2) 구역(zone)/건물 전체 누적 위험도 (세션을 넘어 파일로 이어서 갱신)
를 함께 계산해서
  - 화면에는 "원본 + 균열 마스킹 오버레이" 한 장만 표시하고
  - 위험도 정보는 화면 하단의 별도 박스에 크게 강조해서 표시하고
  - 모든 로그(세션 시작/종료, 확정 균열, 주기적 스냅샷 등)는
    log.txt 파일에도 함께 저장하는 스크립트.

모델 구조/전처리는 기존과 동일:
  - torchvision.models.segmentation.deeplabv3_resnet101 + DeepLabHead(2048, 1)
  - 입력: Resize((img_size, img_size)) -> ToTensor() (0~1 스케일, 정규화 없음)
  - 학습 시 MSELoss + 0/255 범위의 마스크를 그대로 사용했으므로 모델 출력이
    절대값 0~1로 정규화되어 있지 않음 -> 화면 표시 전 min-max 정규화 후
    트랙바로 threshold 조절

위험도 계산 로직은 crack_risk_analysis.py / building_risk_tracker.py 에
분리되어 있음 (각 파일 상단 docstring에 설계 배경 설명).

사용법:
    python webcam_crack_test.py --model seg_model.pth
    python webcam_crack_test.py --model seg_model.pth --camera 0 --img-size 256 \
        --zone "3F_복도" --state-file building_risk_state.json \
        --log-file log.txt --mm-per-pixel 0.35

키 조작:
    q : 종료 (종료 시 세션 요약이 상태 파일 + log.txt에 저장됨)
    s : 현재 화면(위험도 박스 포함) 캡처 저장 (capture_YYYYmmdd_HHMMSS.png)
    z : 현재 구역(zone) 이름 변경 (콘솔에 입력)
    b : 현재까지의 건물 위험도 요약을 콘솔 + log.txt에 출력
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
    """콘솔 + log.txt 파일에 동시에 기록하는 로거를 만든다."""
    logger = logging.getLogger("crack_risk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 재실행/재초기화 시 핸들러 중복 방지

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def build_model(num_classes: int = 1) -> torch.nn.Module:
    """학습 노트북의 get_model()과 동일한 구조. 가중치는 우리가 따로 불러올
    것이므로 backbone pretrained 다운로드는 받지 않음(오프라인에서도 동작)."""
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
    """웹캠 프레임(BGR, HxWx3) -> 정규화된 grayscale 예측 마스크(0~255, img_size x img_size)"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))

    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    output = model(tensor)["out"][0, 0].detach().cpu().numpy()
    mask_norm = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mask_norm


def make_overlay(frame_bgr, binary_mask_u8):
    """crack으로 판정된 픽셀을 빨간색으로 반투명 오버레이 (원본 위에 마스킹만 표시)"""
    overlay = frame_bgr.copy()
    overlay[binary_mask_u8 > 0] = (0, 0, 255)  # BGR: 빨강
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
    """화면 하단에 위험도 전용 박스를 그려서 위험도 정보만 크고 명확하게 표시.
    - 박스 테두리 색은 건물 전체 등급 색으로 표시해서 한눈에 위험 수준을 파악 가능
    - 1번째 줄: 이번 프레임 위험도(현재 보고 있는 지점)
    - 2번째 줄: 이 구역 누적 최댓값 + 건물 전체 누적 위험도
    """
    h, w = img.shape[:2]
    box_h = max(80, min(140, int(h * 0.22)))
    box_top = h - box_h

    # 반투명 검정 배경 박스
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

    line1 = (f"[{zone}]  Frame risk: {frame_result['frame_score']:.0f} ({frame_grade})  "
              f"cracks={frame_result['num_cracks']}   FPS:{fps:.1f}")
    line2 = f"{zone_text}    |    Building overall: {building_state.data['overall_score']:.0f} ({ov_grade})"

    font_scale1 = max(0.5, min(0.75, box_h / 150))
    font_scale2 = max(0.6, min(0.9, box_h / 120))

    y1 = box_top + int(box_h * 0.42)
    y2 = box_top + int(box_h * 0.82)

    cv2.putText(img, line1, (14, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale1, frame_color, 2, cv2.LINE_AA)
    cv2.putText(img, line2, (14, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale2, ov_color, 2, cv2.LINE_AA)

    # 건물 전체 등급 색으로 화면 테두리를 강조 -> 위험도만 따로 눈에 띄게
    cv2.rectangle(img, (0, box_top), (w - 1, h - 1), ov_color, thickness=3)
    return img


def main():
    parser = argparse.ArgumentParser(description="Webcam crack segmentation + risk assessment")
    parser.add_argument("--model", type=str, default="esw_seg_model_best.pt", help="학습된 모델(.pt) 경로")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 장치 인덱스")
    parser.add_argument("--img-size", type=int, default=256, help="모델 입력 크기 (학습 때와 동일하게)")
    parser.add_argument("--threshold", type=int, default=127, help="초기 threshold 값 (0~255)")
    parser.add_argument("--display-width", type=int, default=800, help="표시 창의 너비(px)")
    parser.add_argument("--cpu", action="store_true", help="GPU가 있어도 강제로 CPU 사용")
    parser.add_argument("--zone", type=str, default="unspecified", help="현재 촬영 중인 구역/방 이름")
    parser.add_argument("--state-file", type=str, default="building_risk_state.json",
                         help="건물 위험도를 세션 간 이어서 저장할 파일 경로")
    parser.add_argument("--mm-per-pixel", type=float, default=None,
                         help="보정값(선택). 있으면 실제 mm 단위 균열 폭 기준으로 등급 산출, "
                              "없으면 픽셀 기준 상대 심각도를 사용")
    parser.add_argument("--log-file", type=str, default="log.txt",
                         help="위험도 산출 로그를 저장할 파일 경로")
    parser.add_argument("--log-interval", type=float, default=5.0,
                         help="새로 확정된 균열이 없어도 현재 위험도를 로그에 남기는 주기(초)")
    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from: {args.model}")
    model = load_model(args.model, device, logger)
    logger.info("Model loaded.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (index={args.camera}). 다른 인덱스를 시도해보세요.")

    window_name = "Crack Segmentation - Risk-aware Webcam Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", window_name, args.threshold, 255, lambda x: None)

    building_state = BuildingRiskState(args.state_file)
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    building_state.start_session(session_id)
    building_state.recompute_overall()  # 이전 세션 결과가 있으면 시작부터 화면/로그에 표시

    logger.info(f"Session started | session_id={session_id} zone={args.zone} "
                f"state_file={args.state_file} log_file={args.log_file}")

    frame_tracker = FrameTracker()
    current_zone = args.zone

    prev_time = time.time()
    fps = 0.0
    last_log_time = 0.0

    print("[INFO] 's': 캡처저장 / 'z': 구역(zone) 변경 / 'b': 건물 위험도 요약 출력 / 'q': 종료")

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

            # --- 위험도 분석 ---
            frame_result = analyze_mask(binary_mask, mm_per_pixel=args.mm_per_pixel)
            newly_confirmed = frame_tracker.update(frame_result["cracks"])
            for crack in newly_confirmed:
                building_state.record_confirmed_crack(current_zone, crack, session_id)
                logger.info(
                    f"Confirmed crack | zone={current_zone} track_id={crack['track_id']} "
                    f"score={crack['score']} grade={crack['grade']} "
                    f"width_mm={crack.get('width_mm')} width_px={crack['width_px']:.2f}"
                )
            if newly_confirmed:
                building_state.recompute_overall()

            # 화면에는 원본 + 마스킹 오버레이만 표시 (마스크 단독 패널은 표시하지 않음)
            overlay_bgr = make_overlay(frame, binary_mask)

            # FPS 계산
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # 일정 주기로 현재 위험도 스냅샷을 로그에 남김 (새 균열이 없어도 추이 확인용)
            if now - last_log_time >= args.log_interval:
                z = building_state.data["zones"].get(current_zone)
                zone_max = z["max_score_ever"] if z else 0
                zone_grade = z["max_grade_ever"] if z else "-"
                logger.info(
                    f"Snapshot | zone={current_zone} frame_score={frame_result['frame_score']:.1f} "
                    f"frame_grade={frame_result['frame_grade']} cracks={frame_result['num_cracks']} "
                    f"zone_max={zone_max}({zone_grade}) "
                    f"building_overall={building_state.data['overall_score']}"
                    f"({building_state.data['overall_grade']})"
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
        logger.info(f"State file: {args.state_file} (다음 실행 시 자동으로 이어서 누적됩니다)")


if __name__ == "__main__":
    main()