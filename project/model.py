import torch
import torch.nn as nn
import torchvision.models as models


class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()

        for param in self.cnn.parameters():
            param.requires_grad = False

        for param in self.cnn.layer3.parameters():
            param.requires_grad = True

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.vit = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        )

        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.cnn(x)

        x = self.fc1(x)

        x = x.unsqueeze(1)
        x = self.vit(x)
        x = x.squeeze(1)

        x = self.fc2(x)

        return x