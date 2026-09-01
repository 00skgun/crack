# 기본 실행
libcamerify python raspberrypi_cam_test.py --model seg_model.pth

# 해상도를 더 낮추어 실행 (성능 최적화)
libcamerify python pi_webcam_crack_test.py --model seg_model.pth --cap-width 320 --cap-height 240 --img-size 256