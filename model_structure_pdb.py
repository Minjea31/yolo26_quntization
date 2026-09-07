from ultralytics import YOLO
import torch
import pdb

# 모델 로드
model = YOLO("yolo26n.yaml").model

# 더미 입력 생성
x = torch.zeros(1, 3, 640, 640)
