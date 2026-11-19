"""Loading the AI4I 2020 data, physics-based features and a leakage-free train/test split.

Leakage: the columns TWF, HDF, PWF, OSF and RNF say WHICH failure happened.
"Machine failure" is (almost) the OR of them, so using them as inputs would let
a model read the answer instead of predicting it. They are never used as features.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"

TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]       # leakage: never features
SENSOR_COLUMNS = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
TYPE_COLUMNS = ["Type_L", "Type_M", "Type_H"]
PHYSICS_COLUMNS = ["Temperature difference", "Power", "Strain"]


def load(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(SENSOR_COLUMNS + [TARGET, "Type"]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    return df


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that follow the physics behind the failure modes.

    Temperature difference [K]  process - air temperature (heat dissipation, HDF)
    Power [W]                   torque * angular speed (power failure, PWF)
    Strain [min*Nm]             tool wear * torque (overstrain, OSF)
    """
    out = df.copy()
    out["Temperature difference"] = out["Process temperature"] - out["Air temperature"]
    out["Power"] = out["Torque"] * out["Rotational speed"] * 2.0 * math.pi / 60.0
    out["Strain"] = out["Tool wear"] * out["Torque"]
    return out


def make_xy(df: pd.DataFrame, physics: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix X and label y. Product type is one-hot encoded (L, M, H)."""
    d = add_physics_features(df) if physics else df
    X = d[SENSOR_COLUMNS].copy()
    for t in ("L", "M", "H"):
        X[f"Type_{t}"] = (d["Type"] == t).astype(int)
    if physics:
        X[PHYSICS_COLUMNS] = d[PHYSICS_COLUMNS]
    y = d[TARGET].astype(int)
    return X, y


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified split of the raw rows, so both parts keep the 3.4 % failure rate."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    return train.reset_index(drop=True), test.reset_index(drop=True)
