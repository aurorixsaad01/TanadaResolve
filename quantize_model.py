import tensorflow as tf
import pandas as pd
import pickle
import os

print("=== TanadaResolve: INT8 Edge Quantization (V2) ===")

# 1. Loading our trained Keras model and the new unified Preprocessor
print("[1/3] Loading Keras model and preprocessor...")
model = tf.keras.models.load_model("tanada_base_model.keras")

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# 2. Preparing the Representative Dataset
df = pd.read_csv("tanadasynth.csv")
X = df.drop(columns=["node_id", "ml_label"])

# Using our saved pipeline to transform the calibration data instantly
calibration_data = preprocessor.transform(X.head(500)).astype("float32")

def representative_dataset():
    """Feeds data to the converter one row at a time for calibration."""
    for i in range(len(calibration_data)):
        yield [calibration_data[i].reshape(1, -1)]

# 3. Configure the TensorFlow Lite Converter
print("[2/3] Initializing TensorFlow Lite Converter...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8 

# 4. Execute Quantization
print("[3/3] Compressing 32-bit weights to 8-bit integers...")
tflite_quant_model = converter.convert()

output_filename = "tanada_quantized_int8.tflite"
with open(output_filename, "wb") as f:
    f.write(tflite_quant_model)

# 5. Generate the Thesis Memory Optimization Report
keras_size = os.path.getsize("tanada_base_model.keras") / 1024
tflite_size = os.path.getsize(output_filename) / 1024

print("\n=== THESIS: EDGE MEMORY OPTIMIZATION REPORT ===")
print(f"Original Keras Model Size : {keras_size:.2f} KB")
print(f"Quantized INT8 Edge Model : {tflite_size:.2f} KB")
print(f"Compression Ratio         : {keras_size / tflite_size:.1f}x smaller")
print("\n[SUCCESS] Phase 3 Re-Quantization Complete. Ready for Inference!")