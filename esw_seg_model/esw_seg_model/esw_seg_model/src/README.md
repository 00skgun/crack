# Raspberry Pi 4 NCNN 실행

이 실행기는 PyTorch `.pt` 가중치를 로드하지 않습니다. 다음 NCNN 모델 파일 두 개를 함께 사용합니다.

- `esw_seg_model_ncnn/esw_seg_model_fp16_160.ncnn.param`: NCNN 모델 그래프
- `esw_seg_model_ncnn/esw_seg_model_fp16_160.ncnn.bin`: 실제 FP16 가중치

`esw_seg_model_best.pt`는 모델 재변환과 검증을 위한 원본 파일입니다. Raspberry Pi 추론에는 필요하지 않습니다.

## 설치

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-picamera2
python3 -m venv --system-site-packages .venv-ncnn
source .venv-ncnn/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_ncnn_pi4.txt
```

## 기본 실행

CSI 리본 케이블 카메라:

```bash
python webcam_crack_ncnn.py --camera-backend picamera2
```

기본값 `auto`도 Picamera2를 먼저 선택하므로 다음 명령도 같습니다.

```bash
python webcam_crack_ncnn.py
```

USB 웹캠:

```bash
python webcam_crack_ncnn.py --camera-backend opencv --camera 0
```

처리 주기를 가장 빠르게 설정:

```bash
python webcam_crack_ncnn.py --process-every 1 --threads 4
```

발열과 CPU 부하를 줄이는 기본 권장 설정:

```bash
python webcam_crack_ncnn.py --headless --process-every 3 --threads 3 --cap-fps 5
```

자세한 옵션과 모델 정보는 [`esw_seg_model_ncnn/README_PI4.md`](esw_seg_model_ncnn/README_PI4.md)를 참고하세요.
