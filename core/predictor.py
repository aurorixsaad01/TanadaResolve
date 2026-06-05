import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

class EdgeAIPredictor:
    def __init__(self, tflite_path="tanada_quantized_int8.tflite", preprocessor_path="preprocessor.pkl"):
        print("[SYSTEM] Booting Edge AI Inference Engine...")
        
        # 1. Loading the Preprocessor (Scalers & One-Hot Encoders)
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)
            
        # 2. Loading the INT8 TFLite Model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        # The exact labels our neural network learned
        self.class_names = {0: "NOMINAL", 1: "SENSOR_CONTRADICTION", 2: "CASCADE_RISK"}

    def infer(self, sensor_dict):
        """
        Processes a raw dictionary of sensor readings, applies TinyML INT8 quantization,
        and returns the Neural Network's prediction.
        """
        # Step 1: Preprocess the raw dictionary into a flat scaled array
        df = pd.DataFrame([sensor_dict])
        float_input = self.preprocessor.transform(df).astype(np.float32)
        
        # Step 2: Hardware Quantization (Convert Float32 to INT8)
        # This is the exact math an ESP32 uses to compress inputs for the AI
        scale, zero_point = self.input_details['quantization']
        if scale > 0:
            int8_input = np.round((float_input / scale) + zero_point).astype(np.int8)
        else:
            int8_input = float_input.astype(np.int8)
            
        # Step 3: Feed to Neural Network & Invoke
        self.interpreter.set_tensor(self.input_details['index'], int8_input)
        self.interpreter.invoke()
        
        # Step 4: Read Output Probabilities & Decode INT8 back to Float
        output_data = self.interpreter.get_tensor(self.output_details['index'])[0]
        predicted_class = np.argmax(output_data)
        
        # TFLite INT8 outputs must be de-quantized to get real [0.0 - 1.0] probabilities
        out_scale, out_zero = self.output_details['quantization']
        if out_scale > 0:
            probabilities = (output_data.astype(np.float32) - out_zero) * out_scale
        else:
            probabilities = output_data.astype(np.float32)
            
        confidence_pct = probabilities[predicted_class] * 100.0
        
        return self.class_names[predicted_class], round(float(confidence_pct), 2)