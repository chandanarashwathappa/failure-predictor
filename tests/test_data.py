import math

import numpy as np
import pandas as pd
import pytest

from fp import data


@pytest.fixture(scope="module")
def df():
    return data.load()


def test_dataset_shape_and_class_balance(df):
    assert len(df) == 10_000
    assert df[data.TARGET].sum() == 339                 # 3.39 % failures
    assert df.isna().sum().sum() == 0


def test_no_leakage_columns_in_features(df):
    for physics in (False, True):
        X, y = data.make_xy(df, physics)
        assert not set(data.FAILURE_MODES) & set(X.columns), "failure-mode labels must not be features"
        assert data.TARGET not in X.columns
        assert len(X) == len(y)


def test_physics_features():
    row = pd.DataFrame([{"Type": "L", "Air temperature": 300.0, "Process temperature": 309.5,
                         "Rotational speed": 1500, "Torque": 40.0, "Tool wear": 100,
                         "Machine failure": 0}])
    out = data.add_physics_features(row).iloc[0]
    assert out["Temperature difference"] == pytest.approx(9.5)
    assert out["Power"] == pytest.approx(40.0 * 1500 * 2 * math.pi / 60)   # 6283.2 W
    assert out["Strain"] == pytest.approx(4000.0)


def test_type_one_hot_has_fixed_columns_even_if_a_type_is_missing(df):
    X, _ = data.make_xy(df[df["Type"] != "H"], physics=False)
    assert list(X.columns[-3:]) == data.TYPE_COLUMNS
    assert X["Type_H"].sum() == 0
    assert (X[data.TYPE_COLUMNS].sum(axis=1) == 1).all()


def test_split_is_stratified_and_disjoint(df):
    train, test = data.split(df, seed=42)
    assert len(train) == 8000 and len(test) == 2000
    rate = df[data.TARGET].mean()
    assert abs(train[data.TARGET].mean() - rate) < 0.002
    assert abs(test[data.TARGET].mean() - rate) < 0.002


def test_missing_column_is_reported(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Type": ["L"], "Torque": [40]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        data.load(p)
