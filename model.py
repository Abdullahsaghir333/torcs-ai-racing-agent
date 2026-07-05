import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# === Load specified CSVs ===
csv_files = [
    "1.csv", "2.csv", "3.csv", "4.csv", "5.csv",
    "6.csv", "7.csv", "11.csv", "12.csv", "13.csv"
]
# csv_files = [os.path.join("..", "client", fname) for fname in csv_files]

missing = [f for f in csv_files if not os.path.isfile(f)]
if missing:
    print(f"❌ Missing files: {missing}")
    exit()

dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(csv_files)} files with {len(df)} total rows")

# === Strip and deduplicate columns ===
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()]
print("🔍 Columns after cleanup:", df.columns.tolist())

# === Drop unnecessary columns ===
drop_cols = (
    ['CurrentLapTime', 'LastLapTime', 'FuelLevel', 'Damage',
     'DistanceFromStart', 'DistanceCovered', 'RacePosition', 'Z', 'Clutch'] +
    [f'Opponent_{i}' for i in range(1, 37)] +
    [f'Focus_{i}' for i in range(1, 6)]
)
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# === Define input/output columns ===
input_features = (
    ['SpeedX', 'SpeedY', 'SpeedZ', 'Angle', 'RPM', 'Gear', 'TrackPosition'] +
    [f'Track_{i}' for i in range(1, 20)] +
    [f'WheelSpinVelocity_{i}' for i in range(1, 5)]
)
output_targets = ['Acceleration', 'Braking', 'Steering']

# === Check for missing columns ===
missing_columns = [col for col in input_features + output_targets if col not in df.columns]
if missing_columns:
    raise ValueError(f"❌ Missing required columns: {missing_columns}")

# === Drop rows with NaN in inputs/outputs ===
df = df[input_features + output_targets]
df.dropna(inplace=True)
print(f"✅ Cleaned data: {df.shape[0]} rows remain")

# === Prepare features and labels ===
X = df[input_features].copy()
y = df[output_targets].copy()

# === Normalize ===
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === Train ===
model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"✅ Model trained. Test MSE: {mse:.4f}")

# === Save ===
joblib.dump(model, 'torcs_ai_model.pkl')
joblib.dump(scaler_X, 'input_scaler.pkl')
joblib.dump(scaler_y, 'output_scaler.pkl')
print("✅ Model and scalers saved.")
