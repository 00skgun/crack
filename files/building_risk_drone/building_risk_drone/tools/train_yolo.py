"""
[선택 사항] 정확도를 더 높이고 싶다면 YOLOv8을 손상 탐지용 데이터셋으로
파인튜닝할 수 있습니다. 이 스크립트는 템플릿이며,
실제 학습에는 GPU 환경과 라벨링된 데이터셋이 필요합니다.
(예: Roboflow Universe의 concrete-crack / structural-damage 공개 데이터셋)

사용 예:
  pip install ultralytics
  python tools/train_yolo.py --data data.yaml --epochs 50

data.yaml 예시는 README.md의 "정확도를 높이고 싶다면" 섹션을 참고하세요.

학습 후 라즈베리파이에 배포할 때는 .pt를 그대로 쓰지 말고
NCNN 또는 TFLite로 export하는 것을 권장합니다:
  model.export(format="ncnn")   # 또는 format="tflite"
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.yaml 경로 (Roboflow export 등)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--model", default="yolov8n.pt", help="베이스 모델 (경량 모델 권장)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
