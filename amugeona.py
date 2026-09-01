import os
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
import pnnx

# 1. 경로 설정 및 출력 폴더 자동 생성
input_weights = os.path.expanduser('~/Desktop/esw_seg_model_best.pt')
output_dir = os.path.expanduser('~/Desktop/seg_ncnn_output')
os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 새로 생성

# 2. 모델 정의
class CrackSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = deeplabv3_resnet101(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)
        base.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.base = base
        
    def forward(self, x):
        return torch.sigmoid(self.base(x)['out'])

# 3. 가중치 로드
print(f'[*] Loading weights from: {input_weights}')
net = CrackSegNet()
sd = torch.load(input_weights, map_location='cpu', weights_only=True)
net.base.load_state_dict(sd, strict=True)
net.eval()

# 4. TorchScript 변환 및 저장
x = torch.randn(1, 3, 256, 256)
traced_model_path = os.path.join(output_dir, 'crackseg_traced_256.pt')

traced = torch.jit.trace(net, x, strict=False)
traced.save(traced_model_path)
print(f'[*] TorchScript traced and saved to: {traced_model_path}')

# 5. PNNX를 이용한 NCNN 변환
print('[*] Starting PNNX conversion to NCNN (FP16)...')
pnnx.convert(
    traced_model_path,
    inputs=x, 
    fp16=True,
    ncnnparam=os.path.join(output_dir, 'crackseg.ncnn.param'),
    ncnnbin=os.path.join(output_dir, 'crackseg.ncnn.bin'),
    ncnnpy=os.path.join(output_dir, 'c256_ncnn.py'),
    pnnxparam=os.path.join(output_dir, 'c256.pnnx.param'),
    pnnxbin=os.path.join(output_dir, 'c256.pnnx.bin'),
    pnnxpy=os.path.join(output_dir, 'c256_pnnx.py'),
    pnnxonnx=os.path.join(output_dir, 'c256.pnnx.onnx'),
)

print(f'[*] Conversion successful! All files are saved in: {output_dir}')
