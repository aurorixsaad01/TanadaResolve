"""
Biological Automation: Phenological Stage Inference via Growing Degree Days (GDD).
Base temperature for rice (Japonica) is typically 10°C.
"""

def infer_growth_stage(cumulative_gdd):
    """
    Infers the rice phenological stage based on accumulated thermal time.
    Values approximate typical Japanese short-grain rice cycles.
    """
    if cumulative_gdd < 400:
        return "seedling"
    elif cumulative_gdd < 800:
        return "tillering"
    elif cumulative_gdd < 1200:
        return "panicle_initiation"
    elif cumulative_gdd < 1600:
        return "heading"
    elif cumulative_gdd < 2000:
        return "grain_filling"
    else:
        return "maturity"

def calculate_daily_gdd(t_max, t_min, base_temp=10.0):
    """Calculates single-day GDD."""
    mean_temp = (t_max + t_min) / 2.0
    return max(0.0, mean_temp - base_temp)