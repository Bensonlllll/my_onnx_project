import numpy as np
from torchvision import datasets, transforms

# 預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 載入 MNIST 測試資料
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 提取前100筆
images = [test_dataset[i][0].numpy() for i in range(10000)]
labels = [test_dataset[i][1] for i in range(10000)]

# 儲存為 npy 檔
np.save("images.npy", np.stack(images, axis=0).astype(np.float32))  # (100, 1, 28, 28)
np.save("labels.npy", np.array(labels))                             # (100,)
print("已儲存 images.npy 和 labels.npy")
