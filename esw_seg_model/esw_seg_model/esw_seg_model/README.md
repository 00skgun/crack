# Raspberry Pi 4 NCNN Crack Segmentation + 건물 위험도 평가

웹캠 영상에 NCNN으로 변환·최적화한 crack segmentation 모델을 적용해서
1) 균열을 화면에 마스킹하고
2) 균열의 폭/길이/면적을 측정해 위험도(A~E 등급)를 계산하고
3) 건물을 돌아다니며(여러 구역/여러 세션) 탐색한 결과를 파일에 누적해서
   구역별·건물 전체 위험도를 계속 이어서 갱신하는 스크립트입니다.

> ⚠️ **중요**: 이 도구가 산출하는 위험도는 자동화된 스크리닝 보조 지표입니다.
> 실제 건물 안전 판정은 반드시 자격을 갖춘 구조기술사 등 전문가의
> 정밀 점검으로 확인해야 합니다.

---

## 1. 파일 구성

| 파일 | 역할 |
|---|---|
| `src/webcam_crack_ncnn.py` | Raspberry Pi 4용 실행 진입점. 카메라 캡처, NCNN 추론, 위험도 오버레이 |
| `src/crack_pump_main.py` | 균열을 연속 확인한 뒤 GPIO 17/27로 펌프를 안전하게 펄스 구동하는 메인 |
| `src/esw_seg_model_ncnn/esw_seg_model_fp16_160.ncnn.param` | NCNN 모델 그래프 |
| `src/esw_seg_model_ncnn/esw_seg_model_fp16_160.ncnn.bin` | 실제 실행에 사용하는 FP16 NCNN 가중치 |
| `src/crack_risk_analysis.py` | 마스크에서 균열별 폭/길이/면적과 등급(A~E)을 계산하는 모듈 |
| `src/building_risk_tracker.py` | 동일 균열 추적과 구역/건물 위험도 누적 저장 모듈 |
| `src/requirements_ncnn_pi4.txt` | Raspberry Pi 4 NCNN 실행 패키지 목록 |
| `src/esw_seg_model_best.pt` | 원본 PyTorch 가중치. 재변환·검증용이며 Pi 실행 시 로드하지 않음 |

> Raspberry Pi에서는 `.pt` 파일이 아니라 `.ncnn.param`과 `.ncnn.bin` 두 파일을 함께 사용합니다.
> 기본 경로가 실행기에 설정되어 있으므로 별도의 모델 인자를 주지 않아도 됩니다.

---

## 2. 설치

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-picamera2 python3-gpiozero

cd src
python3 -m venv --system-site-packages .venv-ncnn
source .venv-ncnn/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_ncnn_pi4.txt
```

- 추론은 Raspberry Pi 4 CPU에서 NCNN으로 실행됩니다. PyTorch와 CUDA는 필요하지 않습니다.
- 모델 입력은 변환 시 고정된 `160×160`이며 실행 옵션으로 바꾸지 않습니다.

---

## 3. 실행 방법

### 기본 실행

```bash
cd src
python webcam_crack_ncnn.py
```

CSI 리본 케이블 카메라를 명시적으로 선택하려면 다음처럼 실행합니다.

```bash
python webcam_crack_ncnn.py --camera-backend picamera2
```

기본값 `auto`도 Picamera2를 먼저 선택합니다. USB 웹캠은
`--camera-backend opencv`를 사용합니다.

### 균열 감지 + 펌프 실행

먼저 GPIO 출력 없이 화면과 감지 조건을 시험합니다.

```bash
python crack_pump_main.py --camera-backend picamera2 --dry-run
```

정상 동작을 확인한 다음 실제 펌프를 실행합니다.

```bash
python crack_pump_main.py --camera-backend picamera2 \
    --pump-in3-pin 17 \
    --pump-in4-pin 27 \
    --min-crack-area-percent 3.0 \
    --confirm-frames 3 \
    --pump-seconds 2.0
```

기본 동작은 균열 면적이 3% 이상인 상태가 추론 3회 연속 확인되면 펌프를 2초간
한 번 구동하는 방식입니다. 같은 균열이 계속 보일 때 반복 구동하지 않으며, 균열이
3회 연속 사라진 후에만 다음 작동을 대기합니다.

> 펌프나 모터를 GPIO에 직접 연결하지 마세요. 반드시 별도 전원과 모터 드라이버를
> 사용하고 Raspberry Pi와 드라이버의 GND를 공통으로 연결하세요. GPIO 번호는 BCM 기준입니다.

### 옵션을 활용한 실행 예시

```bash
python webcam_crack_ncnn.py \
    --camera 0 \
    --cap-width 320 \
    --cap-height 240 \
    --cap-fps 5 \
    --process-every 3 \
    --threads 4 \
    --zone "3F_복도" \
    --state-file building_risk_state.json \
    --mm-per-pixel 0.35
```

### 옵션 설명

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--param` | `esw_seg_model_ncnn/esw_seg_model_fp16_160.ncnn.param` | NCNN 모델 그래프 경로 |
| `--bin` | `esw_seg_model_ncnn/esw_seg_model_fp16_160.ncnn.bin` | FP16 NCNN 가중치 경로 |
| `--camera-backend` | `auto` | `auto`, CSI용 `picamera2`, USB용 `opencv` 중 선택 |
| `--camera` | `0` | 카메라 인덱스 (여러 개면 0,1,2... 시도) |
| `--cap-width` | `320` | 카메라 캡처 너비(px) |
| `--cap-height` | `240` | 카메라 캡처 높이(px) |
| `--cap-fps` | `5` | 카메라 요청 FPS |
| `--process-every` | `3` | 지정한 프레임 수마다 한 번 추론. `1`이면 매 프레임 추론 |
| `--threads` | 최대 `4` | NCNN 추론 스레드 수 |
| `--threshold` | `127` | 이진화 임계값 초기값 (실행 중 화면의 트랙바로 실시간 조절 가능) |
| `--display-width` | `640` | 화면에 표시되는 창의 너비(px) |
| `--roi` | (없음) | 집중 분석 영역을 `x,y,width,height` 형식으로 지정 |
| `--headless` | (off) | 화면 표시 없이 처리 성능을 우선하는 모드 |
| `--zone` | `unspecified` | 현재 촬영 중인 구역/방 이름. 실행 중 `z` 키로도 변경 가능 |
| `--state-file` | `building_risk_state.json` | 건물 위험도 누적 상태를 저장할 파일 경로. **이 파일을 유지해야 실행할 때마다 위험도가 이어서 누적됩니다** |
| `--mm-per-pixel` | (없음) | 보정값. 있으면 균열 폭을 실제 mm 단위로 계산해서 등급을 매김. 없으면 픽셀 기준 상대 심각도 사용 (4번 항목 참고) |
| `--log-file` | `log_ncnn.txt` | 위험도 산출 로그(세션 시작/종료, 확정 균열, 주기적 스냅샷 등)를 저장할 파일 경로 |
| `--log-interval` | `5.0` | 새로 확정된 균열이 없어도 현재 위험도를 로그에 남기는 주기(초) |

---

## 4. `--mm-per-pixel` 보정하는 방법 (선택, 하지만 권장)

보정 없이도 실행은 되지만, 이 경우 위험도는 "픽셀 기준 상대 심각도"로만
표시되어 절대적인 mm 폭과 다를 수 있습니다. 정확한 mm 단위 등급을 얻으려면:

1. 카메라를 실제 촬영할 위치(벽면과의 거리)에 고정
2. 크기를 아는 물체(예: 가로 85.6mm 신용카드, 또는 줄자)를 벽면에 대고 한 프레임 캡처
3. 그 물체가 화면에서 차지하는 픽셀 폭을 확인 (그림판, Python `cv2.imshow` 등으로 확인 가능)
4. `mm_per_pixel = 실제 물체 폭(mm) / 화면 속 픽셀 폭`
5. 이 값을 `--mm-per-pixel`에 넣어 실행

카메라-벽면 거리가 촬영마다 크게 달라지면 보정값도 매번 달라지므로,
가능하면 일정한 거리에서 촬영하는 것을 권장합니다.

---

## 5. 실행 중 키 조작

| 키 | 동작 |
|---|---|
| `q` | 종료 (종료 시 이번 세션 요약이 `--state-file`에 자동 저장됨) |
| `s` | 현재 화면(3분할 패널) 캡처 저장 (`capture_YYYYmmdd_HHMMSS.png`) |
| `z` | 구역(zone) 이름 변경 — 콘솔에 새 이름 입력 (다른 방/층으로 이동할 때 사용) |
| `b` | 지금까지 누적된 건물 전체 위험도 요약을 콘솔에 출력 |

화면에는 **원본 영상 + 균열 마스킹(빨간색 반투명 오버레이)** 한 장만 표시됩니다.
위험도 정보는 화면 **하단의 별도 박스**에 크게 표시되어, 마스킹된 영상과
위험도 수치를 한눈에 구분해서 볼 수 있습니다. 박스 테두리 색은 건물 전체
등급 색으로 표시되어(초록=A ~ 빨강=E) 위험 수준을 색으로도 바로 파악할 수 있습니다.

```
┌───────────────────────────────────────────────────────────┐
│ [3F_복도]  Frame risk: 42 (C)  cracks=2   FPS:24.3          │
│ Zone max: 68(D) *WIDENING*   |   Building overall: 55 (C)   │
└───────────────────────────────────────────────────────────┘
```

- 1번째 줄: 지금 이 프레임에서 감지된 균열 중 가장 위험한 것의 점수/등급, 균열 개수, FPS
- 2번째 줄(왼쪽): 이 구역에서 지금까지(이전 실행 포함) 관측된 최댓값. `*WIDENING*`이 붙으면
  직전 세션 대비 균열이 커지고 있다는 뜻(진행성 균열 의심, 우선 점검 필요)
- 2번째 줄(오른쪽): 모든 구역을 종합한 건물 전체 위험도

### log_ncnn.txt 로그

같은 위험도 정보가 화면뿐 아니라 `--log-file`(기본 `log_ncnn.txt`)에도 시간순으로 저장됩니다.
아래 이벤트들이 기록됩니다.

| 이벤트 | 기록 시점 |
|---|---|
| `Session started` | 프로그램 시작 시 |
| `Confirmed crack` | 새로운 균열이 확정(연속 프레임 검출)될 때마다 |
| `Snapshot` | `--log-interval`(기본 5초)마다, 새 균열이 없어도 현재 위험도 기록 |
| `Zone changed` | `z` 키로 구역을 변경했을 때 |
| `Saved capture` | `s` 키로 화면을 캡처했을 때 |
| `Manual summary requested` | `b` 키를 눌렀을 때 |
| `Session ended` | `q` 키로 종료할 때, 최종 요약 |

예시:
```
2026-08-24 05:22:47 [INFO] Session started | session_id=20260824_052247_ab12cd zone=3F_복도 state_file=building_risk_state.json log_file=log_ncnn.txt
2026-08-24 05:22:52 [INFO] Confirmed crack | zone=3F_복도 track_id=a1b2c3d4 score=75 grade=D width_mm=1.42 width_px=4.73
2026-08-24 05:22:57 [INFO] Snapshot | zone=3F_복도 frame_score=42.0 frame_grade=C cracks=2 zone_max=75(D) building_overall=61.5(D)
```

---

## 6. 건물 위험도가 누적되는 방식 (요약)

1. **프레임 내 중복 방지**: 같은 균열이 여러 프레임에 계속 잡혀도, 연속 3프레임 이상 검출된
   경우에만 "확정"으로 인정하고 세션당 1회만 카운트 (단발성 오탐 배제 + 위험도 폭주 방지)
2. **구역별 누적**: 확정된 균열은 현재 `zone`의 통계(최댓값, 개수)에 반영되고
   `--state-file`에 저장되어 다음 실행에도 이어짐
3. **진행성 판단**: 같은 구역을 다시 탐색했을 때 직전 세션보다 균열이 확연히(20%↑) 커졌으면
   `growth_flag`가 켜지고 건물 전체 점수에 가산점(+15) 부여
4. **건물 전체 점수** = `0.6 × 가장 위험한 구역 점수 + 0.3 × 전체 구역 평균 점수 + (진행성 발견 시 +15)`
   — 평균만 쓰면 위험한 구역 하나가 희석되므로 최댓값 비중을 높게 둠

자세한 설계 배경과 한계는 `building_risk_tracker.py` 상단 docstring에
자세히 적어두었습니다 (특히 "마커/측위 없이는 개별 균열을 세션 너머로
완벽히 재식별할 수 없다"는 한계는 실사용 전 꼭 읽어보시길 권합니다).

---

## 7. 위험도 등급 기준 커스터마이징

`crack_risk_analysis.py`의 `WIDTH_GRADE_TABLE_MM`에 예시 등급표가 있습니다.
실제 프로젝트에 적용할 때는 해당 구조물에 적용되는 공식 기준
(예: 국토안전관리원 정밀안전점검 지침, 관련 콘크리트구조 기준 등)의
균열폭 등급 구간으로 반드시 교체해서 사용하세요.

```python
# crack_risk_analysis.py
WIDTH_GRADE_TABLE_MM = [
    (0.1, "A", 5),
    (0.3, "B", 25),
    (1.0, "C", 50),
    (2.0, "D", 75),
]
WIDTH_GRADE_MAX = ("E", 100)
```

건물 전체 점수 산식(가중치 0.6/0.3, 가산점 15, 진행 판정 임계값 20%)은
`building_risk_tracker.py` 상단의 상수(`GROWTH_RATIO_THRESHOLD`)와
`recompute_overall()` 함수에서 조정할 수 있습니다.

---

## 8. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| CSI 카메라에서 프레임을 읽지 못함 | `rpicam-hello`로 카메라를 확인하고 `sudo apt install -y python3-picamera2` 실행 후 `--camera-backend picamera2` 사용 |
| USB 카메라를 열 수 없음 | `--camera-backend opencv`와 `--camera` 인덱스 0,1,2를 시도. 다른 프로그램의 카메라 점유 여부 확인 |
| NCNN 모델을 열 수 없음 | `.ncnn.param`과 `.ncnn.bin`이 모두 `src/esw_seg_model_ncnn/`에 있는지 확인. Git LFS로 받은 `.bin`이 포인터 파일이면 `git lfs pull` 실행 |
| 균열 폭 측정이 부정확함 | `scikit-image` 설치 여부 확인(`pip install scikit-image`) — 미설치 시 근사값 사용됨. 그래도 부정확하면 `--mm-per-pixel` 보정 여부 확인 |
| 위험도가 매 프레임 계속 올라감 | `--state-file`은 "구역의 최댓값"을 저장하는 것이라 이미 최댓값을 넘지 않는 한 더 오르지 않음. 계속 오른다면 실제로 더 큰/새로운 균열이 잡히고 있는 것 |
| `state-file`을 초기화하고 싶음 | 해당 JSON 파일을 삭제하고 다시 실행하면 새로 시작됨 |

---

## 9. 한계 및 주의사항

- 단일(모노큘러) 웹캠에는 절대 스케일 정보가 없어 `--mm-per-pixel` 보정 없이는
  실제 mm 단위 폭이 아닌 상대 심각도만 제공됩니다.
- 마커나 실내 측위 없이는 서로 다른 세션(재방문)에서 "정확히 동일한 물리적 균열"을
  영상만으로 재식별할 수 없습니다. 그래서 개별 균열이 아닌 **구역 단위**로
  최댓값 추이를 추적하는 방식을 사용합니다. 더 정밀한 개별 균열 추적이 필요하면
  고정 카메라 위치, QR/AR 마커, 실내측위(UWB 등), SLAM 기반 맵핑 도입을 검토하세요.
- 이 스크립트의 위험도는 스크리닝 보조 지표이며, 최종 안전 판정은 전문가의
  현장 정밀점검을 통해 이루어져야 합니다.
