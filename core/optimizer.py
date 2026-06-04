import numpy as np

def quantize_to_float32(raw_matrix):
    """
    Reduces the meomory footprint of incoming 64-bit telemetry matrices
    by reducing them to 32-bit floats for Edge AI microcontrollers.
    """
    if not isinstance(raw_matrix, np.ndarray):
        raise TypeError("Input must be a valid NumPy ndarray.")
    
    return raw_matrix.astype(np.float32)

def calculate_memory_savings(raw_matrix):
    """
    Returns a dictionary detailing the exact byte reduction
    acheved via optimization.
    """
    original_bytes = raw_matrix.nbytes
    optimized_bytes = quantize_to_float32(raw_matrix).nbytes
    saved_bytes = original_bytes - optimized_bytes

    return {
        "original_kb": original_bytes / 1024.0,
        "optimized_kb": optimized_bytes / 1024.0,
        "saved_kb": saved_bytes / 1024.0
    }
