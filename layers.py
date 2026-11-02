import torch
import torch.nn.functional as F
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),#b,32,128,128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),#32,64,64

            nn.Conv2d(32, 64, 3, padding=1),#64,64,64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),#64,32,32

            nn.Conv2d(64, 128, 3, padding=1),#128,32,32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),#256,16,16

            nn.AdaptiveAvgPool2d((8, 8)),#B,256,8,8
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(2)#B,256,64
        x = x.transpose(1, 2)#B,64,256
        return x

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear#128.64
        self.r = r
        self.scaling = alpha / r  #y = Wx + b + (alpha/r)*BAx

        in_features = base_linear.in_features#64,128
        out_features = base_linear.out_features

        for p in self.base.parameters():
            p.requires_grad = False

        self.A = nn.Parameter(
            torch.randn(
                r, in_features,
                device=base_linear.weight.device,
                dtype=torch.float32
            ) * 0.01
        )#8,128

        self.B = nn.Parameter(
            torch.zeros(
                out_features, r,
                device=base_linear.weight.device,
                dtype=torch.float32
            )
        )#64,8


    def forward(self, x):#N, 128
        base_out = self.base(x)#Wx+b

        x_lora = x.to(torch.float32)

        lora_out = (x_lora @ self.A.T) @ self.B.T
        lora_out = lora_out * self.scaling
        return base_out + lora_out.to(base_out.dtype)