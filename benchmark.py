import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from core.predictor import EdgeAIPredictor
import warnings

# Suppress TFLite deprecation warnings for clean terminal output
warnings.filterwarnings('ignore')

print("=== TanadaResolve: Thesis Benchmark Evaluation ===")
print("[SYSTEM] Pitting Single-Sensor Baseline vs. 5-Sensor Edge AI...\n")

# 1. Load the exact same unseen Test Set used in Phase 2
df = pd.read_csv("tanadasynth.csv")
X = df.drop(columns=["ml_label", "node_id"])
y = df["ml_label"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Label mapping for readable output
label_map = {0: "NOMINAL", 1: "SENSOR_CONTRADICTION", 2: "CASCADE_RISK"}
y_true_strings = [label_map[val] for val in y_test]

# 2. Recreate the Legacy Single-Sensor Baseline Logic (engine.py equivalent)
STAGE_THRESHOLDS = {
    "seedling": {"min": 25.0, "max": 35.0},
    "tillering": {"min": 5.0, "max": 15.0},
    "panicle_initiation": {"min": 20.0, "max": 30.0},
    "heading": {"min": 25.0, "max": 35.0},
    "grain_filling": {"min": 15.0, "max": 25.0},
    "maturity": {"min": 5.0, "max": 15.0}
}

def legacy_baseline_predict(row):
    """Simulates a standard irrigation system that only checks moisture thresholds."""
    moisture = row['moisture_pct']
    stage = row['growth_stage']
    
    bounds = STAGE_THRESHOLDS[stage]
    
    if moisture < bounds['min']:
        return "SENSOR_CONTRADICTION" # Assumes drought/sensor break
    elif moisture > bounds['max']:
        return "CASCADE_RISK"         # Assumes flooding
    else:
        return "NOMINAL"

# 3. Execute the Race
baseline_predictions = []
ai_predictions = []

print("[PROCESSING] Booting EdgeAIPredictor (INT8 TFLite Model)...")
ai_engine = EdgeAIPredictor(tflite_path="tanada_quantized_int8.tflite", preprocessor_path="preprocessor.pkl")

print(f"[PROCESSING] Running {len(X_test)} unseen test samples through both systems...\n")
for i in range(len(X_test)):
    row_data = X_test.iloc[i].to_dict()
    
    # Baseline Prediction
    baseline_predictions.append(legacy_baseline_predict(row_data))
    
    # Edge AI Prediction
    ai_predictions.append(ai_engine.infer(row_data)[0])

# 4. Generate the Final Academic Tables
print("==========================================================")
print("   SYSTEM 1: LEGACY SINGLE-SENSOR BASELINE (engine.py)    ")
print("==========================================================")
print(f"Accuracy: {accuracy_score(y_true_strings, baseline_predictions) * 100:.2f}%\n")
print(classification_report(y_true_strings, baseline_predictions, digits=3))

print("\n==========================================================")
print("   SYSTEM 2: TINYML 5-SENSOR FUSION AI (predictor.py)     ")
print("==========================================================")
print(f"Accuracy: {accuracy_score(y_true_strings, ai_predictions) * 100:.2f}%\n")
print(classification_report(y_true_strings, ai_predictions, digits=3))

print("\n[SUCCESS] Benchmark complete. Empirical evidence generated.")