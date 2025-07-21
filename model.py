import torch
import torch.nn as nn

class OFDMNet_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64 ,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                     # 🔽 加 Dropout（可以調整）
            nn.Linear(64 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 55 * 2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 64, 5) -> (B, 5, 64)
        x = self.cnn(x)
        x = self.fc(x)
        return x.view(-1, 55, 2)

class OFDMNet_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 55 * 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(x.size(0), -1)
        out = self.fc(lstm_out)
        return out.view(-1, 55, 2)

class OFDMNet_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc = nn.Linear(5, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Sequential(
            nn.Linear(128 * 64, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 55 * 2)
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, 55, 2)