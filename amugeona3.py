import cv2
import numpy as np
import ncnn
import time
from gpiozero import DigitalOutputDevice

# ==========================================
# 하드웨어 설정 (펌프)
# ==========================================
in3 = DigitalOutputDevice(17)
in4 = DigitalOutputDevice(27)

def pump_on():
    in3.on()
    in4.off()

def pump_off():
    in3.off()
    in4.off()

pump_off()

# ==========================================
# NCNN 모델 설정
# ==========================================
net = ncnn.Net()
# net.opt.use_vulkan_compute = True  # 라즈베리파이 가속 가능시 주석 해제

net.load_param("crackseg.ncnn.param")
net.load_model("crackseg.ncnn.bin")

# ==========================================
# 위험도 산출 기준 설정 (누락되었던 변수 추가)
# ==========================================
CRACK_RISK_THRESHOLD = 3.0  # 전체 화면 중 균열이 3% 이상일 때 펌프 가동

# ==========================================
# 카메라 및 메인 루프 설정
# ==========================================
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("[error] cannot open camera.")
    exit()

print("[*] system on... (종료: Ctrl+C)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[error] failed to grab frame from camera.")
            break
            
        # 전처리: 256x256 리사이즈
        img_resized = cv2.resize(frame, (256, 256))
        
        # NCNN Mat 변환
        mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 256, 256)
        
        # 정규화
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        
        # 추론
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")
        
        # 후처리 및 위험도 계산
        out_array = np.array(mat_out).reshape(256, 256)
        crack_mask = out_array > 0.5 
        
        crack_pixels = np.sum(crack_mask)
        total_pixels = 256 * 256
        crack_ratio = (crack_pixels / total_pixels) * 100.0
        
        # 펌프 제어
        if crack_ratio >= CRACK_RISK_THRESHOLD:
            print(f"[DANGER!] crack: {crack_ratio:.2f}% -> pump on!")
            pump_on()
        else:
            print(f"[normal] crack: {crack_ratio:.2f}% -> pump waiting")
            pump_off()
            
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[*] Exit by user.")

finally:
    pump_off()
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Pump off and resources released.")
