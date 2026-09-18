"""Convert the trained PyTorch model to ONNX format for lightweight deployment."""
import os
import torch
import torch.nn as nn

# Same CNN architecture as app.py
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Load trained weights
model = CNN()
model.load_state_dict(torch.load("models/cifar10_cnn.pth", map_location="cpu"))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 32, 32)
os.makedirs("models", exist_ok=True)
onnx_path = "models/cifar10_cnn.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

size_kb = os.path.getsize(onnx_path) / 1024
print(f"ONNX model saved to {onnx_path} ({size_kb:.0f} KB)")
