"""
LEGACY COMPONENT: V1 Rule-Based Baseline Engine
Note: This file contains the original 2-stage if-statement logic. 
It is preserved here in the legacy/ folder to serve as the "baseline comparison system" 
referenced in the thesis abstract. The active production system now uses the INT8 ML model via core/predictor.py.
"""
import logging
import numpy as np
import pandas as pd

# Creating a root level logger orchestrator
logger = logging.getLogger("TanadaEngine")
logger.setLevel(logging.INFO) # Set lowest severity barrier

# Creating a handler meant strictly for saving errors to disk
file_handler = logging.FileHandler("paddy_system.log")
file_handler.setLevel(logging.ERROR) # ONLY capture serious bugs/corruptions in the file

# Creating a format to stamp timestamps and severity levels automatically
log_format = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
file_handler.setFormatter(log_format)

# Attach the handler to your main engine logger
logger.addHandler(file_handler)

class TanadaEngine:
    def __init__(self, sensor_labels):
        self.labels = sensor_labels

        # New state attributes: Operational Metric Accumulators
        self.total_matrices_processed = 0
        self.total_anomalies_flagged = 0
        self.total_corruptions_blocked = 0
        
        # FIX 1: Saved my rule dictionary to 'self' so all functions can read it!
        self.rules = {
            "flooding": {
                "min_moisture": 15.0
            },
            "drainage": {
                "min_moisture": 5.0
            }
        }

    # FIX 2: Added growth_stage to the method signature, defaulting to "flooding"
    def resolve_contradictions(self, raw_matrix, growth_stage="flooding"):
        """
        Converts raw multi-sensor matrices into DataFrames and flags anomalies based on growth stages.
        """
        # Cleanly look up the moisture threshold from our saved rules dictionary
        selected_rules = self.rules.get(growth_stage, self.rules["flooding"])
        moisture_threshold = selected_rules["min_moisture"]

        if raw_matrix.shape[1] != len(self.labels):
            logger.error("Ingested matrix column layout is structurally corrupted.")
            self.total_corruptions_blocked += 1
            return None
        
        self.total_matrices_processed +=1

        # Convert incoming array to a DataFrame wrapper
        df = pd.DataFrame(data=raw_matrix, columns=self.labels)

        # Vectorized check using my dynamic threshold variable
        contradiction_condition = (df["moisture_pct"] < moisture_threshold) & (df["acoustic_velocity"] > 1540.0)

        # Count anamolies in this batch and update the counter
        batch_anomalies = contradiction_condition.sum()
        self.total_anomalies_flagged += batch_anomalies

        # Build out statuses
        df["resolution_status"] = "NOMINAL"
        df.loc[contradiction_condition, "resolution_status"] = "RESOLVE: SENSOR_CONTRADICTION"

        return df
    
    def get_analytics_summary(self):
        """
        Returns a dictionary containing the current state of operational metrics.
        """
        return {
            "total_matrices_processed": self.total_matrices_processed,
            "total_anomalies_flagged": self.total_anomalies_flagged,
            "total_corruptions_blocked": self.total_corruptions_blocked
        }
     