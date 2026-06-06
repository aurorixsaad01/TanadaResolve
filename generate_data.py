from core.generators import generate_tanadasynth

print("Initializing TanadaSynth Generation...")

# Generating 10,000 rows of physics-bound telemetry
df = generate_tanadasynth(10000)

# Save to a CSV file in root folder
df.to_csv("tanadasynth.csv", index=False)

print(f"Successfully generated {len(df)} rows to tanadasynth.csv!")
print("Dataset preview:")
print(df.head())