# ESW 균열 세그멘테이션 NCNN — Raspberry Pi 4

## 구성

- `esw_seg_model_fp16_160.ncnn.param`: NCNN 그래프
- `esw_seg_model_fp16_160.ncnn.bin`: FP16 저장 가중치
- `model_metadata.json`: 원본/변환 모델 해시와 수치 검증 결과
- `../webcam_crack_ncnn.py`: Pi 4용 실시간 카메라 실행 파일

두 NCNN 모델 파일은 항상 같은 디렉터리에 함께 있어야 합니다.

## 모델 특성

- 구조: DeepLabV3 + ResNet-101, 출력 채널 1개
- 입력: RGB `1×3×160×160`, 값 범위 `0~1`
- 출력: `1×1×160×160`
- 변환: PNNX optlevel 2
- 가중치 저장: FP16
- 실행 누산: FP32
- 연산량: 약 47.161 GFLOPs

Raspberry Pi 4의 Cortex-A72는 네이티브 ARMv8.2 FP16 산술을 지원하지 않으므로
FP16은 모델 파일과 가중치 저장 공간을 줄이는 데 사용하고, 계산 누산은 FP32로
유지합니다.

## Raspberry Pi OS 64-bit 설치

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-picamera2 python3-gpiozero

cd crack/esw_seg_model/esw_seg_model/esw_seg_model/src
python3 -m venv --system-site-packages .venv-ncnn
source .venv-ncnn/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_ncnn_pi4.txt
```

## 실행

CSI 리본 케이블 카메라:

```bash
python webcam_crack_ncnn.py --camera-backend picamera2 --zone "1F_로비"
```

기본값 `auto`도 CSI 카메라에서는 Picamera2를 먼저 사용합니다.

```bash
python webcam_crack_ncnn.py --zone "1F_로비"
```

USB 웹캠:

```bash
python webcam_crack_ncnn.py --camera-backend opencv --camera 0
```

## 펌프 연동 메인

실제 GPIO를 사용하기 전에 dry-run으로 감지 조건을 확인합니다.

```bash
python crack_pump_main.py --camera-backend picamera2 --dry-run
```

기존 모터 드라이버 IN3=BCM17, IN4=BCM27 배선으로 실행:

```bash
python crack_pump_main.py --camera-backend picamera2 \
  --pump-in3-pin 17 --pump-in4-pin 27 \
  --min-crack-area-percent 3.0 --confirm-frames 3 --pump-seconds 2.0
```

- 3% 이상 균열 영역을 추론 3회 연속 확인해야 작동합니다.
- 펌프는 2초 후 자동으로 꺼집니다.
- 같은 균열에는 한 번만 작동하고, 균열이 3회 연속 사라져야 다시 대기 상태가 됩니다.
- 종료, 예외, 카메라 오류 시 `finally`에서 펌프를 강제로 끕니다.

> 펌프/모터는 GPIO에 직접 연결하지 말고 반드시 모터 드라이버와 별도 전원을 사용하세요.

모니터 없이 처리량을 우선하는 실행:

```bash
python webcam_crack_ncnn.py --headless --process-every 1 --threads 4
```

발열이나 메모리 부담을 낮추는 실행:

```bash
python webcam_crack_ncnn.py --headless --process-every 3 --threads 3 --cap-fps 5
```

특정 벽 영역만 집중 분석하는 ROI 예시 (`x,y,width,height`):

```bash
python webcam_crack_ncnn.py --roi 80,40,160,160 --process-every 1
```

ROI는 모델 입력 연산량 자체를 더 줄이지는 않지만, 같은 160×160 입력을 관심
영역에 집중시켜 작은 균열의 유효 해상도를 높이는 데 도움이 됩니다.

## 기본 최적화 값

- 카메라: 320×240, 5 FPS
- 추론: 3프레임마다 1회
- NCNN 스레드: 최대 4개
- OpenCV 스레드: 1개
- NCNN light mode 및 packing layout 활성화
- Picamera2 버퍼: 2프레임, OpenCV 버퍼: 1프레임

`--process-every 1`은 탐지 갱신 주기를 가장 빠르게 하지만 DeepLabV3-ResNet101
자체가 무거워 Pi 4에서 수십 FPS는 기대할 수 없습니다. 실제 목표는 지연을 줄인
수 초 주기의 근실시간 모니터링입니다.

## INT8 모델을 포함하지 않은 이유

NCNN의 사후 INT8 양자화는 대표 데이터로 activation calibration이 필요합니다.
공식 가이드는 검증 데이터 5,000장 이상을 권장합니다. 현재 저장소의 샘플 7장만으로
교정하면 균열처럼 얇은 구조의 마스크 정확도가 크게 손상될 수 있어 배포용 INT8은
생성하지 않았습니다. 실제 카메라 환경의 대표 이미지가 충분히 준비되면 별도 INT8
모델을 교정하고 IoU/Dice 재검증 후 사용하는 것이 안전합니다.
