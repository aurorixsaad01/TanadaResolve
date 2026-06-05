import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from core.predictor import EdgeAIPredictor
import warnings
warnings.filterwarnings('ignore')

print("=== TanadaResolve: Feature Importance (Ablation Study) ===")

df = pd.read_csv("tanadasynth.csv")
X = df.drop(columns=["ml_label", "node_id"]).head(1000) # Use 1000 samples for speed
y_true = df["ml_label"].head(1000).map({0: "NOMINAL", 1: "SENSOR_CONTRADICTION", 2: "CASCADE_RISK"})

ai_engine = EdgeAIPredictor()

# Calculate Baseline (All Sensors)
base_preds = [ai_engine.infer(X.iloc[i].to_dict())[0] for i in range(len(X))]
baseline_acc = accuracy_score(y_true, base_preds)
print(f"Baseline Accuracy (All Features): {baseline_acc * 100:.2f}%\n")

# The Sensors to test
sensors = ["moisture_pct", "acoustic_velocity", "ec_ds_m", "temperature_c", "humidity_pct"]

print("Accuracy Drop when Sensor is REMOVED (Zeroed out):")
for sensor in sensors:
    X_ablated = X.copy()
    X_ablated[sensor] = 0.0  # Zero out the sensor to simulate it breaking/missing
    
    ablated_preds = [ai_engine.infer(X_ablated.iloc[i].to_dict())[0] for i in range(len(X))]
    acc = accuracy_score(y_true, ablated_preds)
    drop = baseline_acc - acc
    
    print(f"- {sensor.ljust(20)} : Dropped by {drop * 100:.2f}% (Accuracy: {acc * 100:.2f}%)")