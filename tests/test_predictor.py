import unittest
from core.predictor import EdgeAIPredictor

class TestEdgeAIPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Loads the INT8 hardware model and the preprocessor once before running tests.
        """
        cls.predictor = EdgeAIPredictor(
            tflite_path="tanada_quantized_int8.tflite", 
            preprocessor_path="preprocessor.pkl"
        )

    def test_nominal_condition(self):
        """Asserts the AI correctly identifies perfectly healthy physics."""
        payload = {
            "terrace_elevation": 2, "slope_degree": 14.0, "bund_height_m": 0.50,
            "growth_stage": "heading", "weather_forecast": "sunny",
            "moisture_pct": 30.0, "acoustic_velocity": 1520.0,
            "ec_ds_m": 1.2, "temperature_c": 25.0, "humidity_pct": 60.0
        }
        result = self.predictor.infer(payload)[0]
        self.assertEqual(result, "NOMINAL")

    def test_sensor_contradiction(self):
        """Asserts the AI catches a decoupled sensor (dry moisture + wet acoustics)."""
        payload = {
            "terrace_elevation": 2, "slope_degree": 14.0, "bund_height_m": 0.50,
            "growth_stage": "heading", "weather_forecast": "sunny",
            "moisture_pct": 5.0, "acoustic_velocity": 1555.0,
            "ec_ds_m": 1.2, "temperature_c": 25.0, "humidity_pct": 60.0
        }
        result = self.predictor.infer(payload)[0]
        self.assertEqual(result, "SENSOR_CONTRADICTION")

    def test_thermodynamic_decoupling(self):
        """Asserts the AI catches a cold-temperature + high-velocity physical impossibility."""
        payload = {
            "terrace_elevation": 2, "slope_degree": 14.0, "bund_height_m": 0.50,
            "growth_stage": "heading", "weather_forecast": "sunny",
            "moisture_pct": 28.0, "acoustic_velocity": 1558.0,
            "ec_ds_m": 1.2, "temperature_c": 18.0, "humidity_pct": 60.0
        }
        # The predictor returns a tuple: (Label, Confidence). We grab [0] for the label.
        result = self.predictor.infer(payload)[0]
        self.assertEqual(result, "SENSOR_CONTRADICTION")    

    def test_cascade_risk(self):
        """Asserts the AI catches an upstream Top-Tier flood washing away nutrients."""
        payload = {
            "terrace_elevation": 1, "slope_degree": 12.5, "bund_height_m": 0.45,
            "growth_stage": "heading", "weather_forecast": "heavy_rain",
            "moisture_pct": 45.0, "acoustic_velocity": 1520.0,
            "ec_ds_m": 0.2, "temperature_c": 25.0, "humidity_pct": 60.0
        }
        result = self.predictor.infer(payload)[0]
        self.assertEqual(result, "CASCADE_RISK")

if __name__ == '__main__':
    unittest.main()