"""
webcam_crack_test.py
---------------------------------------------------
학습된 crack segmentation 모델(seg_model.pth)을 웹캠 영상에
실시간으로 적용해서 결과를 확인하는 테스트 스크립트.

모델 구조 / 전처리는 학습 노트북(pytorch-road-segmentation-syncrack.ipynb)과
동일하게 맞췄습니다:
  - torchvision.models.segmentation.deeplabv3_resnet101 + DeepLabHead(2048, 1)
  - 입력: Resize((img_size, img_size)) -> ToTensor() (0~1 스케일, 정규화 없음)
  - 학습 시 MSELoss + 0/255 범위의 마스크를 그대로 사용했기 때문에,
    모델 출력(raw logit)도 절대값이 딱 0~1로 정규화되어 있지 않습니다.
    그래서 화면에 보여줄 때 min-max 정규화 후 트랙바로 threshold를
    직접 조절할 수 있게 만들었습니다. (모델이 잘 수렴했다면 127 근처가
    적당하겠지만, 실사용 웹캠 화질/조명 차이 때문에 조절이 필요할 수 있음)

사용법:
    python webcam_crack_test.py --model seg_model.pth
    python webcam_crack_test.py --model seg_model.pth --camera 0 --img-size 256

키 조작:
    q : 종료
    s : 현재 화면 캡처 저장 (capture_YYYYmmdd_HHMMSS.png)
"""

import argparse
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def build_model(num_classes: int = 1) -> torch.nn.Module:
    """학습 노트북의 get_model()과 동일한 구조. 가중치는 우리가 따로 불러올
    것이므로 backbone pretrained 다운로드는 받지 않음(오프라인에서도 동작)."""
    try:
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=None, weights_backbone=None
        )
    except TypeError:
        # 구버전 torchvision 호환
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)

    model.classifier = DeepLabHead(2048, num_classes)
    return model


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(num_classes=1)
    state_dict = torch.load(weights_path, map_location=device)

    # state_dict가 체크포인트 딕셔너리 형태일 경우 내부 가중치 추출
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # strict=False를 설정하여 aux_classifier 등 추론 시 불필요한 키 무시
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"[INFO] Ignored unexpected keys (e.g. aux_classifier): {unexpected_keys}")

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

    # 모델 raw 출력을 화면 표시용으로 0~255로 min-max 정규화
    mask_norm = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mask_norm


def make_overlay(frame_bgr, binary_mask_u8):
    """crack으로 판정된 픽셀을 빨간색으로 반투명 오버레이"""
    overlay = frame_bgr.copy()
    overlay[binary_mask_u8 > 0] = (0, 0, 255)  # BGR: 빨강
    blended = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)
    return blended


def put_label(img, text):
    cv2.putText(
        img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
    )
    return img


def main():
    parser = argparse.ArgumentParser(description="Webcam crack segmentation live test")
    parser.add_argument("--model", type=str, default="esw_seg_model_best.pt", help="학습된 모델(.pt) 경로")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 장치 인덱스")
    parser.add_argument("--img-size", type=int, default=256, help="모델 입력 크기 (학습 때와 동일하게)")
    parser.add_argument("--threshold", type=int, default=127, help="초기 threshold 값 (0~255)")
    parser.add_argument("--display-width", type=int, default=480, help="각 패널의 표시 너비(px)")
    parser.add_argument("--cpu", action="store_true", help="GPU가 있어도 강제로 CPU 사용")
    args = parser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading model from: {args.model}")
    model = load_model(args.model, device)
    print("[INFO] Model loaded.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (index={args.camera}). 다른 인덱스를 시도해보세요.")

    window_name = "Crack Segmentation - Webcam Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", window_name, args.threshold, 255, lambda x: None)

    prev_time = time.time()
    fps = 0.0

    print("[INFO] 's' 키: 캡처 저장 / 'q' 키: 종료")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 프레임을 읽지 못했습니다. 종료합니다.")
                break

            h, w = frame.shape[:2]

            mask_norm = predict_mask(model, frame, args.img_size, device)
            mask_resized = cv2.resize(mask_norm, (w, h), interpolation=cv2.INTER_LINEAR)

            thresh_val = cv2.getTrackbarPos("Threshold", window_name)
            _, binary_mask = cv2.threshold(mask_resized, thresh_val, 255, cv2.THRESH_BINARY)

            mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            overlay_bgr = make_overlay(frame, binary_mask)

            # FPS 계산
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # 표시용 리사이즈 (세 패널 크기 통일)
            dw = args.display_width
            dh = int(h * dw / w)
            panel_frame = cv2.resize(frame, (dw, dh))
            panel_mask = cv2.resize(mask_bgr, (dw, dh))
            panel_overlay = cv2.resize(overlay_bgr, (dw, dh))

            put_label(panel_frame, f"Camera  FPS:{fps:.1f}")
            put_label(panel_mask, f"Predicted Mask (th={thresh_val})")
            put_label(panel_overlay, "Overlay")

            combined = cv2.hconcat([panel_frame, panel_mask, panel_overlay])
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, combined)
                print(f"[INFO] Saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()