from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Create data directory
raw_dir = Path("../data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# Fetch Seoul Bike Sharing Demand dataset
dataset = fetch_ucirepo(id=560)

# Extract data
X = dataset.data.features
y = dataset.data.targets

# Combine features and target
data = pd.concat([X, y], axis=1)

# Save raw dataset
output_path = raw_dir / "seoul_bike_sharing_demand.csv"
data.to_csv(output_path, index=False)

# Inspect
print(f"Saved to: {output_path}")
print(f"Shape: {data.shape}")
print(data.head())