import torch
import torch.nn as nn
import numpy as np

# 定義模型結構，必須和訓練時完全一樣
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

# 偵測可用的運算裝置（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型檔案和資料檔案的路徑
model_path = "model/mnist_pt_model.pth"
images_path = "images.npy"
labels_path = "labels.npy"

# 載入模型
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 載入資料
images = np.load(images_path)
labels = np.load(labels_path)

# 將 NumPy 資料轉換成 PyTorch Tensor
images_tensor = torch.from_numpy(images).to(device)
labels_tensor = torch.from_numpy(labels).to(device)

# 開始推論
correct = 0
total = labels_tensor.size(0)

# 關閉梯度計算，以提高推論速度
with torch.no_grad():
    outputs = model(images_tensor)
    _, predicted = torch.max(outputs.data, 1)

# 計算總準確率
correct = (predicted == labels_tensor).sum().item()
accuracy = correct / total * 100
print(f"Total: {total} images。")
print(f"Correct: {correct} images。")
print(f"Accuracy: {accuracy:.2f}%")