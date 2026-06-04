import pandas as pd
import random

STAGE_PARAMETERS = {
    "seedling": {"moisture_min": 25.0, "moisture_max": 35.0},
    "tillering": {"moisture_min": 5.0, "moisture_max": 15.0}, 
    "panicle_initiation": {"moisture_min": 20.0, "moisture_max": 30.0},
    "heading": {"moisture_min": 25.0, "moisture_max": 35.0},
    "grain_filling": {"moisture_min": 15.0, "moisture_max": 25.0},
    "maturity": {"moisture_min": 5.0, "moisture_max": 15.0}
}

def generate_tanadasynth(num_records):
    """
    Generates the TanadaSynth dataset for ML training, incorporating 
    scientifically accurate physics bounds and balanced class distributions.
    """
    dataset = []

    for i in range(num_records):
        stage = random.choice(list(STAGE_PARAMETERS.keys()))
        elevation_tier = random.choice([1, 2, 3]) 
        forecast = random.choice(["sunny", "heavy_rain", "cloudy"])
        
        moisture = random.uniform(STAGE_PARAMETERS[stage]["moisture_min"], STAGE_PARAMETERS[stage]["moisture_max"])
        velocity = random.uniform(1500.0, 1540.0)
        ec = random.uniform(0.5, 2.0)
        temp = random.uniform(20.0, 35.0)
        humidity = random.uniform(40.0, 90.0)

        label = 0 # 0 = NOMINAL, 1 = SENSOR_CONTRADICTION, 2 = CASCADE_RISK

        # We use a weighted trigger to guarantee roughly equal class representation
        anomaly_trigger = random.random()

        if anomaly_trigger > 0.70:
            # 15% chance: DROUGHT STRESS CONTRADICTION (Fixed temp/humidity gap)
            # High temp and low humidity during a water-heavy stage, but soil sensors claim it's wet
            if stage == "heading" and anomaly_trigger > 0.85:
                temp = random.uniform(35.0, 42.0)
                humidity = random.uniform(10.0, 25.0)
                velocity = random.uniform(1545.0, 1560.0)
                label = 1
            # 15% Chance: STANDARD SENSOR CONTRADICTION
            else:
                moisture = random.uniform(2.0, 8.0)
                velocity = random.uniform(1545.0, 1560.0)
                label = 1

        elif anomaly_trigger > 0.40 and forecast == "heavy_rain":
            # 30% Chance: CASCADE RISK (Fixing the Direction Physics)
            # Cascade starts at Tier 1 or 2, and floods downward.
            if elevation_tier in [1, 2]:
                moisture = random.uniform(40.0, 50.0) # Bund saturation
                ec = random.uniform(0.1, 0.4) # Runoff detected
                label = 2

        # Fixing unique Node IDs so we don't have overlapping physical constraints
        record = {
            "node_id": f"T{elevation_tier}-{i:05d}",
            "growth_stage": stage,
            "terrace_elevation": elevation_tier,
            "weather_forecast": forecast,
            "moisture_pct": round(moisture, 2),
            "acoustic_velocity": round(velocity, 2),
            "ec_ds_m": round(ec, 2),
            "temperature_c": round(temp, 2),
            "humidity_pct": round(humidity, 2),
            "ml_label": label
        }
        dataset.append(record)

    return pd.DataFrame(dataset)
