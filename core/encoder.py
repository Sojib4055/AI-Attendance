import numpy as np
import torch
import torch.nn as nn

class SimpleIrisEncoderNet(nn.Module):
    """Lightweight CNN encoder.

    This is a stand-in for DeepIrisNet2 so the pipeline can run.
    For a real system, replace this with the official DeepIrisNet2
    architecture and load its pretrained weights.
    """
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        # L2 normalize
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        return x

class IrisEncoder:
    def __init__(self, model_path: str = None, device: str = "cuda", embedding_dim: int = 256):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SimpleIrisEncoderNet(embedding_dim=embedding_dim).to(self.device)
        if model_path is not None and model_path.strip() != "":
            try:
                state = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state)
            except FileNotFoundError:
                # run with random weights if file missing
                pass
        self.model.eval()

    def encode(self, norm_iris):
        """norm_iris: np.ndarray [H, W], float32 in [0,1]"""
        arr = norm_iris.astype(np.float32)
        arr = np.expand_dims(arr, axis=(0, 1))  # [1,1,H,W]
        x = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            emb = self.model(x).cpu().numpy()[0]
        return emb.astype(np.float32)
