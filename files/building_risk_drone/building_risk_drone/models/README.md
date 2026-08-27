# models/

`config.yaml`에서 `detection.use_yolo: true`로 설정했을 때 사용할 커스텀 학습 가중치(.pt)를
이 폴더에 넣으세요. (예: `damage_yolov8n.pt`)

가중치가 없거나 `use_yolo: false`이면 프로그램은 자동으로 고전 영상처리 기반
균열 검출기(`src/crack_detector.py`)로 동작하므로, 이 폴더가 비어 있어도
데모 실행에는 문제가 없습니다.

학습 방법은 `tools/train_yolo.py`와 최상위 `README.md`의
"정확도를 높이고 싶다면" 섹션을 참고하세요.
