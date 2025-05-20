import torch
from torch import nn
import timm

class CLFModel(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        model = timm.create_model("resnetv2_101x1_bit.goog_in21k", pretrained=True)
        self.stem = model.stem
        self.stages0 = model.stages[0]
        self.stages1 = model.stages[1]
        self.stages2 = model.stages[2]
        self.stages3 = model.stages[3]
        self.norm = model.norm
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=2048, out_channels=396, kernel_size=(int(image_size/32), int(image_size/32)), padding=(0, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages0(x)
        low_f = self.stages1(x)

        attention = self.attention(low_f)

        x = self.stages2(low_f)
        x = self.stages3(x)
        high_f = self.norm(x)

        attention = torch.sigmoid(attention)
        attention = nn.functional.interpolate(attention, high_f.shape[2:], mode='bilinear', align_corners=False)

        x = high_f * attention + high_f

        x = self.classifier(x)
        return x