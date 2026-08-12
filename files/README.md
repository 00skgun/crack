# 건물 위험도 판별 드론 모듈 (Building Risk Assessment Module for Drones)

임베디드 소프트웨어 경진대회 자유공모용 데모 프로젝트입니다.

지능형 자율비행 드론이 아니라, **"카메라 입력을 받아 건물의 위험도를 판단하는"
모듈 하나에 집중**한 구현체입니다. 드론에 탑재되는 다른 모듈(비행 제어, 통신 등)과
독립적으로 동작하며, 라즈베리파이 + 카메라만 있으면 단독으로 데모 실행이 가능합니다.

라이브 카메라 없이도 `tools/generate_test_image.py`로 만든 합성 이미지로
즉시 파이프라인을 검증할 수 있습니다.

---

## 1. 이 프로젝트가 푸는 문제

> "그래서, 건물의 위험도를 어떻게 파악하지?"

이 질문에 대한 답이 이 프로젝트의 전부입니다. 접근 방식은 두 갈래입니다.

| | 방식 | 장점 | 단점 |
|---|---|---|---|
| A | 딥러닝 객체 탐지 (YOLO 등) | 정확도 높음, 다양한 손상 유형 학습 가능 | **라벨링된 균열/손상 데이터셋 필요**, 학습 시간/GPU 필요 |
| B | 고전 영상처리 (Canny/컨투어 기반 기하학적 분석) | **학습 데이터 없이 즉시 동작**, 라즈베리파이에서 가볍게 실행 | 조명/노이즈에 상대적으로 민감, 세밀한 손상 유형 구분은 어려움 |

경진대회 일정상 대규모 라벨링 데이터셋을 확보하기 어려운 경우가 많다고 판단해서,
**B(고전 영상처리)를 기본 파이프라인으로 하고, A(YOLO)는 가중치 파일만 넣으면
자동으로 전환되는 선택적 업그레이드 경로**로 설계했습니다.
즉 데모 당일 카메라 앞에 균열이 있는 벽(또는 사진)을 대면 바로 결과가 나옵니다.

---

## 2. 핵심 로직: 위험도는 이렇게 계산합니다

전체 흐름은 `src/risk_engine.py`에 있으며, 아래 6단계로 구성됩니다.

```
[카메라 프레임]
      │
      ▼
① 손상 후보 검출          src/crack_detector.py (또는 src/yolo_detector.py)
   그레이스케일 → CLAHE 대비 강화 → Canny 엣지 → 형태학적 닫힘
   → 컨투어 추출 → "길고 얇은 형태"만 1차 필터링
      │
      ▼
② 특징 추출               src/feature_extractor.py
   각 후보에 대해 길이 / 폭 / 방향(각도) 계산
      │
      ▼
③ 방향 분류                (구조공학적 경험칙 기반)
   수직(vertical)   : 위험도 낮음~중간  (건조수축 등 경미한 원인 多)
   수평(horizontal) : 위험도 중간~높음  (휨=bending 관련)
   대각선(diagonal) : 위험도 높음       (전단=shear 관련, 붕괴 전조로 알려짐)
   교차/X자(branching): 위험도 최고     (전단파괴 패턴, 망상균열)
      │
      ▼
④ 개별 손상 심각도 계산    severity = w1·길이정규화 + w2·폭정규화 + w3·방향가중치  (0~100)
      │
      ▼
⑤ 프레임 전체 점수 계산    raw_score = 0.6·max(severity) + 0.2·mean(severity) + 0.2·손상면적비율
      │
      ▼
⑥ 시간적 안정화 + 등급 매핑  EMA(지수이동평균) + 히스테리시스 → 안전/주의/위험/심각
```

### 왜 이런 방식으로 설계했는가

- **방향(각도)을 위험도의 핵심 축으로 삼은 이유**: 균열의 폭이나 길이만으로는
  위험도를 제대로 판단하기 어렵습니다. 실제 구조 진단에서도 균열의 **방향과 패턴**이
  손상 원인(건조수축, 휨, 전단 등)을 유추하는 중요한 단서로 쓰입니다. 특히
  대각선/X자형 균열은 전단력에 의한 손상으로, 건물 붕괴와 더 밀접하게 연관된
  패턴으로 알려져 있어 가중치를 가장 높게 두었습니다.
  (본 로직은 데모/공모전용 휴리스틱이며, 실제 구조 안전진단을 대체하지 않습니다.
  최종 판단은 반드시 구조 전문가의 확인이 필요합니다.)

- **max + mean + coverage를 함께 쓰는 이유**: max만 쓰면 화면 구석의 작은 균열
  하나가 전체 점수를 왜곡할 수 있고, mean만 쓰면 국소적으로 매우 위험한 손상
  하나가 다수의 경미한 영역에 희석되어 과소평가될 수 있습니다. 세 지표를
  가중합하여 "가장 심각한 곳을 놓치지 않으면서도 전반적인 상태를 반영"하도록
  했습니다.

- **EMA + 히스테리시스로 시간적 안정화를 하는 이유**: 실시간 영상은 프레임마다
  조명, 각도, 노이즈가 미세하게 바뀌어 단일 프레임 점수가 출렁입니다. 등급이
  초당 여러 번 안전↔주의로 깜빡이면 데모 신뢰도가 떨어지므로, 최근 N프레임의
  지수이동평균을 쓰고 등급이 **내려갈 때만** 더 보수적인 여유값(margin)을 적용해
  깜빡임을 억제했습니다. 반대로 등급이 **올라갈 때는 즉시 반영**해 안전을
  우선했습니다.

- **개발 중 실제로 발견/수정한 버그** (테스트로 검증됨):
  1. 대각선(45도) 균열이 축정렬 바운딩박스(`boundingRect`) 기준으로는 가로=세로가
     되어 "길고 얇은 형태" 필터에서 걸러지는 문제 → 회전 바운딩박스(`minAreaRect`)로
     교체하여 해결.
  2. X자형(교차) 균열은 두 선이 만나면서 전체 윤곽의 종횡비가 1에 가까워져 같은
     필터에 걸리는 문제 → 면적충전율(contourArea / minAreaRect 면적)이 매우 낮은
     "성긴 큰 구조물"이라는 별도 특징으로 판정하도록 보완. (X자형은 실제로 가장
     위험한 패턴이므로 놓치면 안 되는 케이스였습니다.)
  3. OpenCV의 `cv2.putText`가 한글을 지원하지 않아 화면 텍스트가 깨지는 문제 →
     PIL 기반 렌더링으로 교체, 한글 폰트가 없는 환경에서는 자동으로 영문 라벨로
     대체되도록 폴백 처리.

---

## 3. 폴더 구조

```
building_risk_drone/
├── main.py                    # 실행 진입점 (카메라/이미지/영상 통합 처리 루프)
├── config.yaml                # 위험도 판별 파라미터 (숫자 조정만으로 튜닝 가능)
├── requirements.txt
├── src/
│   ├── camera.py              # picamera2 / USB웹캠 / 이미지 / 영상 통합 인터페이스
│   ├── crack_detector.py      # 고전 영상처리 기반 균열 후보 검출 (학습 불필요)
│   ├── feature_extractor.py   # 후보 → 길이/폭/방향 특징 추출
│   ├── risk_engine.py         # ★ 핵심: 점수화 + 시간적 안정화 + 등급 매핑
│   ├── visualizer.py          # 결과 오버레이 (한글 렌더링 포함)
│   ├── alert.py                # 위험 등급 이상 시 스냅샷/로그 저장
│   └── yolo_detector.py        # (선택) 커스텀 학습 YOLO 모델 연동, 없으면 자동 폴백
├── tools/
│   ├── generate_test_image.py # 카메라 없이 테스트할 합성 균열 이미지 생성기
│   └── train_yolo.py           # (선택) YOLO 파인튜닝 템플릿
├── models/                     # (선택) 커스텀 학습 가중치(.pt) 위치
├── sample_images/               # 생성된 테스트 이미지 및 결과 예시
└── alerts/                      # 위험 감지 시 자동 저장되는 스냅샷/로그
```

---

## 4. 설치

### 4-1. 라즈베리파이 (실제 데모용)

```bash
# 라즈베리파이 OS (Bookworm 기준) - CSI 카메라 모듈 사용 시
sudo apt update
sudo apt install -y python3-picamera2 fonts-noto-cjk

cd building_risk_drone
python3 -m venv --system-site-packages venv   # picamera2는 시스템 패키지이므로 --system-site-packages 필요
source venv/bin/activate
pip install -r requirements.txt
```

USB 웹캠을 쓴다면 `python3-picamera2` 설치 없이 `pip install -r requirements.txt`만으로 충분합니다.

### 4-2. PC (개발/발표자료 준비용)

```bash
cd building_risk_drone
python3 -m venv venv
source venv/bin/activate        # Windows는 venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. 실행

### 카메라 없이 파이프라인 검증 (합성 이미지)

```bash
# 테스트용 균열 이미지 생성 (4가지 패턴)
python3 tools/generate_test_image.py --pattern diagonal --out sample_images/wall_diagonal_crack.jpg
python3 tools/generate_test_image.py --pattern vertical  --out sample_images/wall_vertical_crack.jpg
python3 tools/generate_test_image.py --pattern x         --out sample_images/wall_x_crack.jpg
python3 tools/generate_test_image.py --pattern none      --out sample_images/wall_normal.jpg

# 위험도 판별 실행 (창 없이 결과만 콘솔 출력)
python3 main.py --source sample_images/wall_diagonal_crack.jpg --headless

# 화면으로 확인하고 싶다면 --headless 제거 (아무 키나 누르면 종료)
python3 main.py --source sample_images/wall_diagonal_crack.jpg
```

### 실제 카메라로 실시간 실행

```bash
# 라즈베리파이 CSI 카메라, 모니터 없이 SSH로 원격 실행
python3 main.py --source picam --headless

# USB 웹캠, 모니터 연결된 상태
python3 main.py --source 0

# 결과 영상을 파일로 저장하며 실행 (발표자료용)
python3 main.py --source picam --headless --save-video outputs/demo.mp4
```

### 실행 결과 예시 (합성 이미지 기준, 본 저장소에서 직접 검증)

| 패턴 | 점수 | 등급 |
|---|---|---|
| 정상 벽면 | 0.0 | 안전 |
| 수직 균열 | 46.0 | 주의 |
| 대각선 균열 | 63.9 | 위험 |
| X자(교차) 균열 | 65.5 | 위험 |

방향에 따라 위험도가 의도한 대로 차등 반영되는 것을 확인했습니다.

---

## 6. config.yaml 파라미터 튜닝 가이드

코드를 건드리지 않고 아래 값만 조정해서 데모 환경(조명, 벽면 재질, 카메라 거리)에
맞게 민감도를 튜닝할 수 있습니다.

| 파라미터 | 설명 | 튜닝 힌트 |
|---|---|---|
| `detection.classical_cv.min_crack_length` | 최소 균열 길이(px) | 오탐(노이즈)이 많으면 값을 올리세요 |
| `detection.classical_cv.min_aspect_ratio` | 길고 얇은 형태 판정 기준 | 너무 낮으면 얼룩/그림자를 균열로 오인 |
| `risk_scoring.orientation_weight` | 방향별 위험 가중치 | 대회 심사 기준에 맞춰 재조정 가능 |
| `risk_scoring.width_danger_px` | 이 폭 이상이면 최대 위험으로 간주 | 카메라-벽 거리가 멀면 값을 낮추세요 |
| `camera.frame_skip` | N프레임마다 1번만 분석 | 라즈베리파이가 느리면 값을 올리세요 (반응성↓, 속도↑) |
| `temporal_smoothing.ema_alpha` | 최근 프레임 민감도 | 클수록 반응은 빠르지만 깜빡임도 증가 |
| `levels.*_max` | 등급 경계값 | 데모 환경에서 실제 점수 분포를 보고 조정 |

---

## 7. 라즈베리파이 실시간 성능 최적화

라즈베리파이(특히 3/4/Zero 2W)에서 실시간성을 확보하기 위한 팁입니다.

1. **해상도를 낮추세요.** `config.yaml`의 `camera.width/height`를 640x480 이하로
   (Zero 2W 등 저사양이면 320x240까지) 낮추면 프레임 처리 속도가 크게 향상됩니다.
2. **`frame_skip`을 활용하세요.** 매 프레임을 분석할 필요는 없습니다. 2~3프레임에
   1번만 분석해도 위험도 판단에는 충분하며, 화면 표시는 매 프레임 갱신됩니다.
3. **YOLO를 쓴다면 반드시 export하세요.** `.pt`(PyTorch) 그대로 라즈베리파이에서
   추론하면 매우 느립니다. 학습 후 다음처럼 경량 포맷으로 변환하세요.
   ```python
   from ultralytics import YOLO
   model = YOLO("runs/detect/train/weights/best.pt")
   model.export(format="ncnn")   # 라즈베리파이 CPU에 최적화된 포맷 (권장)
   # 또는
   model.export(format="tflite", int8=True)  # TFLite + INT8 양자화
   ```
4. **고전 CV 파이프라인은 기본적으로 가볍습니다.** YOLO 없이 `classical_cv`만
   사용하는 기본 설정은 라즈베리파이 4 기준으로 640x480에서 별도 최적화 없이도
   실시간에 가깝게 동작합니다(본 개발 환경 기준 프레임당 처리 약 10ms 내외).

---

## 8. 정확도를 더 높이고 싶다면 (선택 사항)

고전 영상처리 파이프라인은 빠르고 학습 데이터가 필요 없다는 장점이 있지만,
조명이 불균일하거나 벽면 재질이 복잡하면 오탐/누락이 늘어날 수 있습니다.
시간이 허락한다면 다음 순서로 딥러닝 검출기를 추가할 수 있습니다.

1. Roboflow Universe 등에서 공개된 콘크리트/구조물 균열 데이터셋을 검색해
   다운로드 (예: "concrete crack detection", "structural damage detection" 등의
   키워드로 검색하면 다양한 공개 데이터셋을 찾을 수 있습니다).
2. `data.yaml` 형식으로 정리:
   ```yaml
   train: dataset/images/train
   val: dataset/images/val
   nc: 5
   names: ["crack", "spalling", "exposed_rebar", "water_leak", "deformation"]
   ```
3. 학습:
   ```bash
   pip install ultralytics
   python3 tools/train_yolo.py --data data.yaml --epochs 50
   ```
4. 학습된 가중치를 `models/damage_yolov8n.pt`에 두고 `config.yaml`에서
   `detection.use_yolo: true`로 변경 → 이후 별도 코드 수정 없이 자동으로
   YOLO 검출기가 사용됩니다 (`src/yolo_detector.py`가 classical CV와 동일한
   출력 스키마로 맞춰주기 때문입니다).

---

## 9. 한계 및 향후 개선 방향 (발표 시 참고)

정직하게 밝히는 것이 데모 신뢰도에 더 도움이 된다고 판단해 명시합니다.

- **고전 CV 파이프라인은 조명 변화, 얼룩, 그림자에 상대적으로 민감**합니다.
  `min_crack_length`, `min_aspect_ratio` 등으로 어느 정도 억제했지만 완벽하지
  않습니다. 실제 배포 환경에서는 YOLO 기반 검출기로 전환하는 것을 권장합니다.
- **카메라 캘리브레이션을 하지 않아 폭/길이가 픽셀 단위 상대값**입니다. 실제
  물리적 mm 단위 균열 폭을 알아야 한다면 카메라 내부 파라미터 보정과 드론의
  대상까지의 거리 추정(예: 초음파/라이다 센서 융합)이 추가로 필요합니다.
- **방향 기반 위험 가중치는 구조공학의 일반적 경험칙을 단순화한 휴리스틱**입니다.
  실제 구조 안전진단을 대체할 수 없으며, 데모/공모전 목적의 판단 로직임을
  명확히 하는 것이 좋습니다.
- **향후 확장 아이디어**: 열화상 카메라를 추가해 온도 이상(누수/화재 전조) 탐지,
  진동 센서를 융합해 구조적 흔들림 감지, 여러 프레임에 걸친 균열 폭 변화를
  추적해 "진행성 손상" 여부 판단 등.

---

## 10. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| 화면 텍스트가 `?????`로 깨짐 | 한글 폰트 없음 | `sudo apt install -y fonts-noto-cjk` 후 재실행 (폰트 없으면 자동으로 영문 라벨로 대체됨) |
| `picamera2 모듈이 없습니다` 오류 | picamera2 미설치 | `sudo apt install -y python3-picamera2`, venv는 `--system-site-packages`로 생성 |
| 카메라를 열 수 없다는 오류 | 다른 프로세스가 카메라 점유 중이거나 권한 문제 | `sudo usermod -aG video $USER` 후 재로그인, 다른 카메라 앱 종료 |
| 균열이 하나도 검출되지 않음 | 임계값이 벽면 재질에 비해 너무 엄격 | `config.yaml`의 `min_crack_length`, `min_aspect_ratio`를 낮춰서 재시도 |
| YOLO 모드인데 계속 classical CV로 동작 | 가중치 파일 경로가 잘못됨 | `detection.yolo_model_path`가 실제 파일을 가리키는지, `ultralytics`가 설치됐는지 확인 |

---

## 라이선스 / 사용 안내

본 저장소는 경진대회 데모/발표용으로 작성되었습니다. 실제 건물 안전진단에
사용할 경우 반드시 구조 전문가의 검증을 거치시기 바랍니다.
