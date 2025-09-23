import onnxruntime as ort
import numpy as np
import argparse
import sys

# 命令列參數解析器
parser = argparse.ArgumentParser()
parser.add_argument("model_type", choices=["tf", "pt"])
args = parser.parse_args()

# 模型路徑
model_path = f"model/mnist_{args.model_type}_model.onnx"

# 載入模型（若失敗則直接退出）
try:
    session = ort.InferenceSession(model_path)
except Exception:
    sys.exit(1)
    
input_name = session.get_inputs()[0].name

# 載入預處理好的圖片與標籤
images = np.load("images.npy")
labels = np.load("labels.npy")

# 推論
outputs = session.run(None, {input_name: images})
preds = np.argmax(outputs[0], axis=1)

# 準確度計算
correct_count = np.sum(preds == labels)
total_count = len(labels)
accuracy = correct_count / total_count * 100

# 顯示結果
print(f"Total: {total_count} images")
print(f"Correct: {correct_count} images")
print(f"Accuracy：{accuracy:.2f}%")
