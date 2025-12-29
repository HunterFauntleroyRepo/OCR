# src/model.py
import torch
import torch.nn as nn
from src.charset import NUM_CLASSES

class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # [B,64,32,128]
            nn.ReLU(),
            nn.MaxPool2d(2,2),                            # [B,64,16,64]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                            # [B,128,8,32]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),                          # [B,256,4,32]

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1))                           # [B,256,2,32]
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=256*2,   # height*channels
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Final output layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)               # [B, C, H, W]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)    # [B, W, C, H]
        x = x.reshape(b, w, c*h)     # flatten H
        x, _ = self.lstm(x)           # [B, W, 512]
        x = self.fc(x)                # [B, W, num_classes]
        x = x.log_softmax(2)          # log-probabilities for CTC loss
        return x
