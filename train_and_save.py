"""
train_and_save.py
-----------------
Run this script ONCE locally to download the dataset, train the model,
and save best_model.pkl + scaler.pkl into the models/ folder.

Usage:
    python train_and_save.py
"""

import os
import numpy as np
import pandas as pd
import urllib.request
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# ── 1. Download dataset ───────────────────────────────────────────────────────
URL  = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/heart-disease/processed.cleveland.data")
FILE = "Data/heart.csv"

if not os.path.exists(FILE):
    print("Downloading dataset …")
    urllib.request.urlretrieve(URL, FILE)
else:
    print("Dataset already exists, skipping download.")

COLUMNS = ["age","sex","chest_pain","resting_bp","cholesterol",
           "fasting_sugar","rest_ecg","max_hr","angina",
           "st_depression","st_slope","vessels","thal","target"]

df = pd.read_csv(FILE, header=None, names=COLUMNS, na_values="?")
df["target"] = (df["target"] > 0).astype(int)

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
df["vessels"] = df["vessels"].fillna(df["vessels"].median())
df["thal"]    = df["thal"].fillna(df["thal"].median())

for col in ["sex","chest_pain","fasting_sugar","rest_ecg",
            "angina","st_slope","vessels","thal","target"]:
    df[col] = df[col].astype(int)

FEATURES = [c for c in df.columns if c != "target"]
X = df[FEATURES].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler      = MinMaxScaler()
X_train_sc  = scaler.fit_transform(X_train)

# ── 3. Train best model ───────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300, max_depth=8,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42)
model.fit(X_train_sc, y_train)

# ── 4. Save artefacts ─────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model,  "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nSaved: models/best_model.pkl")
print("Saved: models/scaler.pkl")
print("\nFeature order (must match app.py):")
for i, f in enumerate(FEATURES):
    print(f"  {i}: {f}")
