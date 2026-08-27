"""
라즈베리파이 CSI 카메라(picamera2), USB 웹캠(cv2.VideoCapture),
정지 이미지, 동영상 파일을 동일한 인터페이스(read())로 다루기 위한 래퍼.
"""
import os
import cv2


class FrameSource:
    def __init__(self, source, width=640, height=480):
        self.width = width
        self.height = height
        self.mode = None
        self.cap = None
        self.picam = None
        self._single_image = None

        if isinstance(source, str) and source.lower() == "picam":
            self._init_picam()
        elif isinstance(source, str) and os.path.splitext(source)[1].lower() in (
            ".jpg", ".jpeg", ".png", ".bmp"
        ):
            self.mode = "image"
            self._single_image = cv2.imread(source)
            if self._single_image is None:
                raise FileNotFoundError(f"이미지를 열 수 없습니다: {source}")
        else:
            self.mode = "video"
            idx = int(source) if str(source).isdigit() else source
            self.cap = cv2.VideoCapture(idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                raise RuntimeError(f"카메라/영상 소스를 열 수 없습니다: {source}")

    def _init_picam(self):
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            config = self.picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            self.mode = "picam"
        except ImportError as e:
            raise RuntimeError(
                "picamera2 모듈이 없습니다. 라즈베리파이에서 "
                "`sudo apt install -y python3-picamera2` 로 설치하세요."
            ) from e

    def read(self):
        if self.mode == "image":
            return True, self._single_image.copy()
        if self.mode == "picam":
            frame = self.picam.capture_array()
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return self.cap.read()

    def is_static_image(self):
        return self.mode == "image"

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.picam is not None:
            self.picam.stop()
