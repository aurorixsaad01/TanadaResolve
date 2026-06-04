import numpy as np
from core.predictor import EdgeAIPredictor

print("=== TanadaResolve Edge AI Simulator ===")   

# Initialize the Neural Network instead of the old if/else engine
ai_engine = EdgeAIPredictor()

# Our live telemetry state
active_payload = {
    "terrace_elevation": 2,
    "slope_degree": 14.0,      # ADDED
    "bund_height_m": 0.50,     # ADDED
    "growth_stage": "heading",
    "weather_forecast": "sunny",
    "moisture_pct": 30.0,
    "acoustic_velocity": 1520.0,
    "ec_ds_m": 1.2,
    "temperature_c": 25.0,
    "humidity_pct": 60.0
}

while True:
    print(f"\n[INFO] Current Stage: {active_payload['growth_stage'].upper()} | Weather: {active_payload['weather_forecast'].upper()}")
    print("[MENU] 1: Run AI Inference | 2: Inject Sensor Contradiction | 3: Inject Cascade Risk | Type 'exit' to quit")
    choice = input("Enter your choice: ").strip().lower()

    if choice == "exit":
        print("\n[INFO] Exiting TanadaResolve. Goodbye!")
        break

    elif choice == "1":
        print("\n[PROCESSING] Passing live telemetry through INT8 Neural Network...")
        print(f"Data: {active_payload}")
        
        # The AI Inference Call!
        prediction = ai_engine.infer(active_payload)
        
        print(f"\n[AI CLASSIFICATION] >>> {prediction} <<<")

    elif choice == "2":
        print("\n[ATTACK] Injecting a physical impossibility (Drought moisture but flooded acoustic velocity)...")
        active_payload["moisture_pct"] = 5.0
        active_payload["acoustic_velocity"] = 1555.0
        print("[SUCCESS] Payload corrupted. Press 1 to see if the AI catches it.")

    elif choice == "3":
        print("\n[ATTACK] Triggering heavy rain on Tier 1 (Upstream Cascade Risk)...")
        active_payload["terrace_elevation"] = 1
        active_payload["weather_forecast"] = "heavy_rain"
        active_payload["moisture_pct"] = 45.0
        active_payload["ec_ds_m"] = 0.2
        active_payload["slope_degree"] = 12.5 # Added
        active_payload["bund_height_m"] = 0.45 # Added
        print("[SUCCESS] Cascade conditions met. Press 1 to evaluate.")

    else:
        print("\n[ERROR] Invalid option selection.")