import torch
import torch.nn as nn

# 定義跟訓練時一樣的模型架構
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

pytorch_model_path = "model/mnist_pt_model.pth"     # 你的 PyTorch 權重檔案
onnx_output_path = "model/mnist_pt_model.onnx"      # 你想輸出的 ONNX 檔名

model = SimpleCNN()
model.load_state_dict(torch.load(pytorch_model_path))
model.eval()

# 建立一個假的輸入（batch size=1, channel=1, 28x28）
dummy_input = torch.randn(1, 1, 28, 28)

# 匯出 ONNX 模型
torch.onnx.export(
    model,                   # 要轉換的模型
    dummy_input,             # 模型輸入範例
    onnx_output_path,        # ONNX 輸出路徑
    input_names=['input'],   # ONNX 的輸入名稱（可自訂）
    output_names=['output'], # ONNX 的輸出名稱
    dynamic_axes={
        'input': {0: 'batch_size'},  # 讓 batch size 可以變動
        'output': {0: 'batch_size'}
    },
    opset_version=13         # ONNX opset 版本
)
print(f"PyTorch to ONNX model save to: {onnx_output_path}")
