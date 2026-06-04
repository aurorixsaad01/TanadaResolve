import unittest
import numpy as np
from core.engine import TanadaEngine

class TestTanadaEngine(unittest.TestCase):
    def setUp(self):
        """
        Runs automatically before EVERY test to set up fresh parameters.
        """
        self.labels = ["moisture_pct", "acoustic_velocity", "ec_ds_m"]
        self.engine = TanadaEngine(self.labels)

    def test_corrupted_matrix_rejection(self):
        """
        Asserts that an invalid matrix shape is actively blocked by the engine.
        """
        # Invalid 2-column matrix
        bad_matrix = np.array([[25.5, 1540.0]]) 
        
        report = self.engine.resolve_contradictions(bad_matrix)
        
        # We EXPECT the engine to return None because it's corrupted!
        self.assertIsNone(report)

    def test_nominal_processing(self):  # Changed from unittest.TestCase to self
        """
        Asserts that a valid matrix passes through successfully.
        """
        # Creating a valid 1-row matrix.
        test_matrix = np.array([
            [27.5, 1540.0, 1.2]
        ])
        
        # Passing it through self.engine.resolve_contradictions()
        report = self.engine.resolve_contradictions(test_matrix)
        self.assertIsNotNone(report)

if __name__ == '__main__':
    unittest.main()
