import tensorflow as tf
from tensorflow.keras import layers, models

# 設定使用 channel 優先 (NCHW) 格式
tf.keras.backend.set_image_data_format('channels_first')

# 設定參數
batch_size = 64
num_epochs = 5

# 載入 MNIST 數據
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 預處理：標準化與升維 (N, C, H, W)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[:, tf.newaxis, :, :]  # (60000, 1, 28, 28)
x_test = x_test[:, tf.newaxis, :, :]

# 建立 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(1, 28, 28)),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)

# 評估
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy：{test_acc * 100:.2f}%")

# 儲存模型
model_path = "model/mnist_tf_model.h5"
model.save(model_path)
print(f"✅ TensorFlow model save to：{model_path}")