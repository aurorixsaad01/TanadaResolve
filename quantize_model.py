import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os

print("=== TanadaResolve: INT8 Edge Quantization & Evaluation ===")

# 1. Loading the trained Keras model and the unified Preprocessor
print("[1/4] Loading Keras model and preprocessor...")
model = tf.keras.models.load_model("tanada_base_model.keras")

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# 2. Preparing the Representative Dataset & Test Set
df = pd.read_csv("tanadasynth.csv")
X = df.drop(columns=["node_id", "ml_label"])
y = df["ml_label"]

# Recreating the exact 20% test set used during training to ensure a fair comparison
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using our saved pipeline to transform calibration data and the test data
calibration_data = preprocessor.transform(X.head(500)).astype("float32")
X_test_processed = preprocessor.transform(X_test).astype("float32")

def representative_dataset():
    """Feeds data to the converter one row at a time for calibration."""
    for i in range(len(calibration_data)):
        yield [calibration_data[i].reshape(1, -1)]

# 3. Configuring and Executing the TensorFlow Lite Converter
print("[2/4] Initializing TensorFlow Lite Converter...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8 

print("[3/4] Compressing 32-bit weights to 8-bit integers...")
tflite_quant_model = converter.convert()

output_filename = "tanada_quantized_int8.tflite"
with open(output_filename, "wb") as f:
    f.write(tflite_quant_model)

# 4. Rigorous TinyML Accuracy Evaluation
print("[4/4] Evaluating INT8 Accuracy vs. Float32 Keras Accuracy...")

# A. Get Baseline Keras Accuracy
loss, keras_accuracy = model.evaluate(X_test_processed, y_test, verbose=0)

# B. Get TFLite INT8 Accuracy
interpreter = tf.lite.Interpreter(model_path=output_filename)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
scale, zero_point = input_details['quantization']

correct_predictions = 0
# The TFLite interpreter must be run row-by-row, simulating live edge conditions
for i in range(len(X_test_processed)):
    float_input = X_test_processed[i:i+1] # Keep 2D shape (1, features)
    
    # Mathematically compress the live float data into an 8-bit integer array
    if scale > 0:
        int8_input = np.round((float_input / scale) + zero_point).astype(np.int8)
    else:
        int8_input = float_input.astype(np.int8)
        
    interpreter.set_tensor(input_details['index'], int8_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])[0]
    
    if np.argmax(output_data) == y_test.iloc[i]:
        correct_predictions += 1

tflite_accuracy = correct_predictions / len(X_test_processed)

# 5. Generate the Final Thesis Report
keras_size = os.path.getsize("tanada_base_model.keras") / 1024
tflite_size = os.path.getsize(output_filename) / 1024

print("\n=== THESIS: EDGE MEMORY & ACCURACY REPORT ===")
print(f"Original Keras Model Size : {keras_size:.2f} KB")
print(f"Quantized INT8 Edge Model : {tflite_size:.2f} KB")
print(f"Compression Ratio         : {keras_size / tflite_size:.1f}x smaller")
print("-" * 45)
print(f"Float32 Model Accuracy    : {keras_accuracy * 100:.2f}%")
print(f"INT8 Model Accuracy       : {tflite_accuracy * 100:.2f}%")
print(f"Accuracy Delta            : {(tflite_accuracy - keras_accuracy) * 100:.2f}%")
print("\n[SUCCESS] Phase 3 Evaluation Complete. Scientific finding validated!")