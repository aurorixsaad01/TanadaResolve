import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
import pickle

print("=== TanadaResolve: Rigorous Neural Network Pipeline ===")

# 1. Loading our generated dataset
print("[1/6] Loading TanadaSynth dataset...")
df = pd.read_csv("tanadasynth.csv")

# 2. Feature Engineering & Train/Test Split
print("[2/6] Separating features and splitting data...")
X = df.drop(columns=["node_id", "ml_label"])
y = df["ml_label"]

# Split FIRST to prevent data leakage during scaling/encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. The ColumnTransformer (Fixed the Encoding Trap)
print("[3/6] Applying One-Hot Encoding and Standard Scaling...")
categorical_features = ["growth_stage", "weather_forecast"]
numerical_features = ["terrace_elevation", "moisture_pct", "acoustic_velocity", "ec_ds_m", "temperature_c", "humidity_pct"]

# This bundles our preprocessing into one clean pipeline we can save later
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# Training the preprocessor on training data, then transforming both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Calculate Class Weights to ensure minority classes are treated equally
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(weights))

# 4. Building the TinyML-Capable Keras Architecture
print("[4/6] Compiling the Keras Sequential MLP Architecture...")
model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Training the Model (with Class Weights!)
print("\n=== STARTING TRAINING EPOCHS ===")
history = model.fit(
    X_train_processed, y_train, 
    epochs=15, 
    batch_size=32, 
    validation_split=0.2,
    class_weight=class_weights_dict,
    verbose=1
)

# 6. Final Evaluation
print("\n=== ACADEMIC EVALUATION (UNSEEN TEST SET) ===")
loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"\n[FINAL SCORE] Overall Neural Network Accuracy: {accuracy * 100:.2f}%\n")

print("=== PER-CLASS PERFORMANCE METRICS ===")
# Predict probabilities, then grab the highest probability class
y_pred_probs = model.predict(X_test_processed, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

target_names = ["NOMINAL (0)", "CONTRADICTION (1)", "CASCADE_RISK (2)"]
print(classification_report(y_test, y_pred, target_names=target_names))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

# 7. Save the Artifacts for Edge Quantization & Inference
model.save("tanada_base_model.keras")
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
print("\n[SUCCESS] Saved 'tanada_base_model.keras' and 'preprocessor.pkl'.")