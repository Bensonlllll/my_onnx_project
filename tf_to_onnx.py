import tf2onnx
import tensorflow as tf

tensorflow_model_path = ("model/mnist_tf_model.h5")
onnx_output_path = "model/mnist_tf_model.onnx"

# 載入 TensorFlow 模型
model = tf.keras.models.load_model(tensorflow_model_path)

# 將模型轉成 ONNX 格式
spec = (tf.TensorSpec((None, 1, 28, 28), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(onnx_output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"✅ TensorFlow to ONNX model save to: {onnx_output_path}")
