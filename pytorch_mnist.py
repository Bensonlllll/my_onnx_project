import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# GPU 優先
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定參數
batch_size = 64
learning_rate = 0.001
num_epochs = 5

transform = transforms.ToTensor()

# 下載資料集
train_dataset = torchvision.datasets.MNIST(root='./pt_data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./pt_data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定義 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3), # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 14x14 → 7x7
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
print("開始訓練...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 測試模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"測試準確率：{accuracy:.2f}%")

# 儲存模型參數
torch.save(model.state_dict(), "model/mnist_pt_model.pth")
print("模型已儲存為 mnist_pt_model.pth")
