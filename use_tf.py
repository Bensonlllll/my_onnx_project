import tensorflow as tf
import numpy as np
import os

# 設定 channel 格式一致（和訓練時一樣）
tf.keras.backend.set_image_data_format('channels_first')

# 模型與資料的路徑
model_path = "model/mnist_tf_model.h5"
images_path = "images.npy"
labels_path = "labels.npy"

# 載入模型
model = tf.keras.models.load_model(model_path)

# 載入圖片與標籤
images = np.load(images_path)  # shape: (N, 1, 28, 28)
labels = np.load(labels_path)  # shape: (N,)

# 確保資料型別正確
images = images.astype(np.float32)
labels = labels.astype(np.int64)

# 推論
logits = model.predict(images, verbose=0)
preds = np.argmax(logits, axis=1)

# 印出詳細預測結果
correct = 0
for i in range(len(images)):
    is_correct = preds[i] == labels[i]
    if is_correct:
        correct += 1

# 準確率
total = len(labels)
accuracy = correct / total * 100
print("TensorFlow: ")
print(f"Total: {total} images")
print(f"Correct: {correct} images")
print(f"Accuracy: {accuracy:.2f}%")
