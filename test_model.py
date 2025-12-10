import torch
from models.unet_resnet18 import build_model

model = build_model()
x = torch.randn(1, 1, 256, 256)
y = model(x)

print("Salida:", y.shape)
