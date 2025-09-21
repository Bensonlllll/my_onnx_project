import onnxruntime as ort
import numpy as np

# 載入 ONNX 模型
session = ort.InferenceSession("model/mnist_pt_model.onnx")
input_name = session.get_inputs()[0].name

# 載入預處理好的圖片與標籤
images = np.load("images.npy")   # (100, 1, 28, 28)
labels = np.load("labels.npy")   # (100,)

# 推論
outputs = session.run(None, {input_name: images})
preds = np.argmax(outputs[0], axis=1)

# 顯示結果
print("預測結果：", preds)
print("正確標籤：", labels)

accuracy = np.mean(preds == labels) * 100
print(f"準確率：{accuracy:.2f}%")
