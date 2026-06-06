# TanadaResolve

**Multi-Sensor Edge-AI Contradiction Resolution for Terraced Rice Paddy Irrigation**

> A TinyML research project targeting offline, cloud-free irrigation intelligence for Obasute Tanada terraced paddy systems in Nagano Prefecture, Japan. Built as proof-of-work toward a MEXT University Recommendation scholarship application for Shinshu University's Graduate School of Agriculture.

---

## Research Context

Traditional smart irrigation systems rely on either cloud connectivity or single-sensor threshold logic — neither of which is viable in the terrain-variable, low-connectivity environment of Japan's mountain terraced paddies (*Tanada*, 棚田). When a dielectric moisture sensor reads *dry* while a co-located acoustic velocity sensor reads *saturated*, a conventional threshold system has no resolution strategy. It fails silently.

TanadaResolve addresses this directly. It fuses five environmental sensors under a contradiction resolution architecture — trained on a physics-grounded synthetic dataset and compressed to 3.6 KB of INT8 weights — to classify incoming telemetry as **NOMINAL**, **SENSOR_CONTRADICTION**, or **CASCADE_RISK** entirely offline, on an ESP32-class microcontroller, with no cloud or network dependency.

The research targets the specific hydraulic challenges of cascading terrace systems: upstream bund saturation, inter-terrace runoff propagation, and sensor failure modes unique to slope-variable soil profiles.

---

## Architecture Overview

The project is structured as a three-phase pipeline:

```
Phase 1 — TanadaSynth Dataset Generation
    └─ generators.py  →  tanadasynth.csv  (10,000 rows, 10 features, 3 classes)

Phase 2 — Neural Network Training
    └─ train_model.py  →  tanada_base_model.keras  +  preprocessor.pkl

Phase 3 — INT8 Quantization & Edge Deployment
    └─ quantize_model.py  →  tanada_quantized_int8.tflite  (3.6 KB)
                         →  firmware/tanada_inference.ino  (ESP32 scaffold)
```

Each phase is independently reproducible. All artifacts are version-controlled except project tree lists tanada_base_model.keras and preprocessor.pkl these are re-generated locally after cloning by running the three phases.

---

## Dataset: TanadaSynth

TanadaSynth is a physics-grounded synthetic telemetry dataset generated to address real-world data scarcity in terraced paddy sensor research.

### Sensor Features

| Feature | Unit | Physical Basis |
|---|---|---|
| `moisture_pct` | % | Volumetric soil moisture (dielectric sensor) |
| `acoustic_velocity` | m/s | Acoustic wave propagation in saturated soil |
| `ec_ds_m` | dS/m | Electrical conductivity (nutrient concentration) |
| `temperature_c` | °C | Ambient temperature |
| `humidity_pct` | % | Relative humidity |
| `terrace_elevation` | tier 1–3 | Cascade position (1=top, 3=bottom) |
| `slope_degree` | ° | Terrace slope angle |
| `bund_height_m` | m | Earthen bund retention height |
| `growth_stage` | categorical | Rice phenological stage (6 stages) |
| `weather_forecast` | categorical | Forecast condition (3 states) |

### Geometric Grounding

Terrace geometry parameters are sourced from published Tanada survey literature:

- **Slope and ridge parameters** — Hamano et al. (2023), *Precision Agriculture*: "Development of a method for detecting the planting and ridge areas in paddy fields using AI, GIS, and precise DEM"
- **Topographical references** — Uchikawa et al. (2018), *International Journal of GEOMATE*
- **Specific slope calibration** — Chikuma City agricultural records and Tanada Society (棚田学会) Obasute survey publications

| Elevation Tier | Slope (°) | Bund Height (m) | Position |
|---|---|---|---|
| Tier 1 | 12.5 | 0.45 | Top (upstream) |
| Tier 2 | 14.0 | 0.50 | Mid-cascade |
| Tier 3 | 15.5 | 0.65 | Bottom (runoff receiver) |

### Thermodynamic Velocity Coupling

Acoustic wave velocity in saturated soil is physically coupled to temperature. NOMINAL records use the empirical relationship:

```
v = 1450.0 + (4.21 × T) − (0.037 × T²)
```

This produces physically realistic velocity ranges (1,519–1,552 m/s for 20–35°C). Anomaly records intentionally violate this coupling — the decoupling itself is the detection signal.

### Class Distribution

| Class | Label | Records | % |
|---|---|---|---|
| NOMINAL | 0 | ~6,300 | 63% |
| SENSOR_CONTRADICTION | 1 | ~3,000 | 30% |
| CASCADE_RISK | 2 | ~700 | 7% |

Gaussian noise (σ: 2% moisture, 8 m/s velocity, 0.05 dS/m EC, 1.5°C temp, 3% humidity) is applied to all base readings to simulate real sensor drift. This creates genuine boundary ambiguity between classes — the classes are not cleanly separable, and the model must learn multi-variate disambiguation.

### Contradiction Types

**SENSOR_CONTRADICTION (Label 1):** Two subtypes:
- *Standard*: Anomalously low moisture (2–8%) with high acoustic velocity (1,545–1,560 m/s) — physically impossible co-occurrence
- *Drought Stress* (heading stage only): High temperature (35–42°C) + low humidity (10–25%) with elevated velocity

**CASCADE_RISK (Label 2):** Heavy rainfall on Tier 1 or Tier 2 — upstream bund saturation (moisture 40–50%) with EC dilution from runoff (0.1–0.4 dS/m). Cascade logic is directionally correct: risk originates upstream, not at the bottom.

---

## Model Architecture

A compact MLP designed for TinyML constraints:

```
Input (17 features after One-Hot Encoding)
    └─ Dense(16, ReLU)
    └─ Dense(8, ReLU)
    └─ Dense(3, Softmax)   ← NOMINAL / SENSOR_CONTRADICTION / CASCADE_RISK
```

**Preprocessing pipeline** (saved as `preprocessor.pkl`):
- `StandardScaler` on all 8 numerical features
- `OneHotEncoder` on `growth_stage` (6 categories) and `weather_forecast` (3 categories)
- `ColumnTransformer` wrapping both — a single saved artifact handles all inference preprocessing

**Training configuration:**
- 80/20 stratified train/test split (`random_state=42`)
- `class_weight='balanced'` — CASCADE_RISK weighted ~5.0× to prevent minority class collapse
- 15 epochs, Adam optimizer, `sparse_categorical_crossentropy`
- Preprocessing fitted on training data only (no data leakage)

---

## Results

### Classification Performance (Unseen Test Set, n=2,000)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| NOMINAL | 1.00 | 0.97 | 0.98 |
| SENSOR_CONTRADICTION | 0.93 | 0.99 | 0.96 |
| **CASCADE_RISK** | **1.00** | **1.00** | **1.00** |
| **Overall Accuracy** | | | **97.55%** |

The 43 NOMINAL→CONTRADICTION misclassifications represent genuine boundary ambiguity: samples where Gaussian noise pushed a normal velocity reading into the contradiction zone. This is the expected behavior of a realistic model, not a defect.

### INT8 Edge Quantization

| Metric | Value |
|---|---|
| Float32 Keras model | 30.23 KB |
| INT8 TFLite model | **3.60 KB** |
| Compression ratio | **8.4×** |
| Float32 accuracy | 97.55% |
| INT8 accuracy | 97.50% |
| Accuracy delta | −0.05% |

The 0.05% accuracy cost of INT8 quantization is within the acceptable tolerance for embedded deployment. The model fits in the flash of an ESP32 with 26.6 KB to spare.

### Feature Importance (Ablation Study)

Accuracy drop when each sensor is zeroed out (1000-sample subset):

| Sensor | Accuracy Drop | Interpretation |
|---|---|---|
| `moisture_pct` | **−62.6%** | Dominant discriminator |
| `ec_ds_m` | −12.4% | CASCADE_RISK key signal (nutrient dilution) |
| `acoustic_velocity` | −3.4% | Contradiction boundary feature |
| `humidity_pct` | −3.2% | Drought stress signal |
| `temperature_c` | +0.6% | Low direct impact in current form |

`temperature_c` shows a slight accuracy increase when removed because the model relies on `acoustic_velocity` as the primary contradiction signal. After StandardScaler normalization, zeroing temperature to 0.0 eliminates a feature that in NOMINAL records is positively correlated with velocity (via the thermodynamic coupling). This removes mild co-correlation noise from the input and slightly clarifies the contradiction boundary. The primary diagnostic value of temperature is indirect — it defines what acoustic velocity *should* be, which the model learns as a background constraint rather than a direct discriminator.

### Baseline Comparison

"benchmark.py implements a 3-class moisture-threshold baseline inspired by engine.py's logic, extended to include CASCADE_RISK detection.":

```
System 1 — Legacy Moisture Threshold (Single Sensor):
    Accuracy:  ~72%
    CASCADE_RISK recall: ~45%  (misses >50% of bund saturation events)

System 2 — TinyML 5-Sensor Fusion (TanadaResolve):
    Accuracy:  97.55%
    CASCADE_RISK recall: 100%  (zero missed upstream flood events)
```

---

## Project Structure

```
TanadaResolve/
│
├── core/                         # Core Python package
│   ├── generators.py             # TanadaSynth physics-grounded data generation
│   ├── gdd.py                    # Growing Degree Day stage inference (Japonica rice)
│   ├── optimizer.py              # Float64→Float32 data buffer optimization
│   ├── predictor.py              # EdgeAIPredictor: INT8 TFLite inference engine
│   ├── __init__.py
│   └── legacy/
│       └── engine.py             # Original rule-based contradiction engine (archived)
│
├── firmware/
│   └── tanada_inference.ino      # ESP32 TFLite Micro inference scaffold
│
├── tests/
│   └── test_predictor.py         # Unit tests for EdgeAIPredictor (4 test cases)
│
├── generate_data.py              # Phase 1: Run TanadaSynth generation → tanadasynth.csv
├── train_model.py                # Phase 2: Train MLP → tanada_base_model.keras
├── quantize_model.py             # Phase 3: INT8 quantize → tanada_quantized_int8.tflite
├── simulator.py                  # Interactive inference terminal (4 options: infer, inject, cascade, GDD stage)
├── benchmark.py                  # Legacy vs. AI accuracy head-to-head comparison
├── ablation_study.py             # Per-sensor feature importance measurement
│
├── tanadasynth.csv               # Generated dataset (10,000 rows)
├── tanada_base_model.keras       # Trained Float32 model (30.2 KB)
├── tanada_quantized_int8.tflite  # Quantized edge model (3.6 KB)
├── preprocessor.pkl              # Saved ColumnTransformer (scaler + encoder)
│
├── pyproject.toml                # Project metadata and dependencies
└── README.md
```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/aurorixsaad01/TanadaResolve.git
cd TanadaResolve
uv sync
```

**Dependencies:** Python ≥ 3.11, TensorFlow ≥ 2.21, scikit-learn ≥ 1.9, pandas ≥ 3.0, numpy ≥ 2.4

---

## Usage

Run the three phases in sequence to reproduce all results from scratch:

```bash
# Phase 1 — Generate the TanadaSynth dataset
python generate_data.py

# Phase 2 — Train the neural network
python train_model.py

# Phase 3 — Quantize to INT8 and evaluate edge accuracy
python quantize_model.py
```

**Run the interactive simulator:**
```bash
python simulator.py
# Option 1: Run AI inference on current telemetry payload (returns class + confidence%)
# Option 2: Inject a sensor contradiction (dry soil + flooded acoustics)
# Option 3: Inject a cascade risk (upstream Tier 1 saturation + EC dilution)
# Option 4: Update growth stage from cumulative GDD input (biological automation)
```

**Run the benchmark (AI vs. legacy threshold system):**
```bash
python benchmark.py
```

**Run the feature ablation study:**
```bash
python ablation_study.py
```

**Run the unit test suite:**
```bash
python -m pytest tests/ -v
```

---

## Biological Automation: Growing Degree Days

`core/gdd.py` provides phenological stage inference from accumulated thermal time, enabling automatic growth stage detection from temperature sensor history rather than manual configuration:

```python
from core.gdd import calculate_daily_gdd, infer_growth_stage

daily_gdd = calculate_daily_gdd(t_max=29.5, t_min=18.0)  # base_temp=10°C for Japonica
cumulative_gdd += daily_gdd
stage = infer_growth_stage(cumulative_gdd)  # → "heading"
```

GDD thresholds are calibrated for Japanese short-grain (Japonica) rice varieties typical of Nagano Prefecture highland cultivation.

---

## ESP32 Deployment

The quantized `.tflite` model is deployable on ESP32 hardware via TensorFlow Lite for Microcontrollers. The firmware scaffold is in `firmware/tanada_inference.ino`.

**Deployment steps:**
1. Convert the `.tflite` binary to a C byte array: `xxd -i tanada_quantized_int8.tflite > model_data.h`
2. Place `model_data.h` in the firmware directory
3. Flash via Arduino IDE with `TensorFlowLite_ESP32` library
4. Wire physical sensors (dielectric moisture, acoustic transducer, EC probe, DHT22) to ESP32 GPIO/I2C pins and replace the dummy input values in `loop()` with real sensor reads

**Memory budget:**
- Model weights: 3.60 KB
- Tensor arena: 10 KB (allocated)
- Remaining ESP32 flash (4 MB typical): 4,086+ KB available

---

## Documented Limitations

**Synthetic data only:** TanadaSynth was generated with physics-grounded parameters from published Tanada survey literature. No real-world sensor deployment has occurred. Model generalisation to actual field conditions at Obasute or Inakura is untested — this is the primary open research question and a target for future lab collaboration.

**GDD requires cumulative manual input:** The simulator's Option 4 infers growth stage from cumulative GDD, but this requires the operator to supply the total accumulated GDD since transplanting. The system does not yet automatically accumulate GDD from a continuous temperature sensor log. Full biological automation would require a persistent temperature history store on the ESP32.

**Firmware is a hardware scaffold:** `tanada_inference.ino` demonstrates the complete TFLM initialisation sequence but does not yet include physical I2C/ADC sensor reads, input quantisation arithmetic in `loop()`, or relay/alarm output logic. `AllOpsResolver` should be replaced with `MicroMutableOpResolver<3>` before deployment to recover ~40–60 KB of flash. The `.tflite` model must also be converted to a C byte array via `xxd -i tanada_quantized_int8.tflite > model_data.h` before flashing.

---

## Research Alignment

This project targets the graduate research agenda of Shinshu University's Faculty of Agriculture (Ina Campus):

- **Prof. UCHIKAWA Yoshiyuki** (Rural Planning Laboratory) — terraced paddy conservation, cascade hydraulic management, Obasute Tanada field studies, land abandonment prevention
- **Prof. HAMANO Mitsuru** — AI + GIS + DEM for paddy ridge detection and hilly-area precision agriculture

TanadaResolve is intended as a proof-of-work baseline demonstrating edge-AI design capability, multi-sensor physics reasoning, and synthetic dataset methodology — positioned as a "Prepared Novice" portfolio piece for MEXT University Recommendation 2027 enrollment.

---

## Acknowledgements

Geometric parameters informed by:
- Hamano et al. (2023), *Precision Agriculture* — paddy ridge detection via AI, GIS, DEM
- Uchikawa et al. (2018), *International Journal of GEOMATE* — slope stability and topographical references
- Chikuma City agricultural records and Tanada Society (棚田学会) Obasute field survey publications

---

## License

MIT License — open for academic use, extension, and adaptation.

---

*TanadaResolve v0.1.0 — Saad Ansari, 2026*
