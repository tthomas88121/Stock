import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split

from config import FEATURE_COLUMNS, MERGED_DATASET_PATH, MODEL_PATH, REG_MODEL_PATH


def train_model():
    start_time = time.time()

    if not Path(MERGED_DATASET_PATH).exists():
        raise FileNotFoundError(f"Merged dataset not found: {MERGED_DATASET_PATH}")

    print("Loading merged dataset...")
    df = pd.read_csv(MERGED_DATASET_PATH)
    print(f"Dataset shape: {df.shape}")

    required_cols = FEATURE_COLUMNS + ["Target", "Target_Return"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in merged dataset: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y_cls = df["Target"].copy()
    y_reg = df["Target_Return"].copy()

    X = X.replace([float("inf"), float("-inf")], pd.NA)
    merged = pd.concat([X, y_cls, y_reg], axis=1).dropna()

    X = merged[FEATURE_COLUMNS].copy()
    y_cls = merged["Target"].copy()
    y_reg = merged["Target_Return"].clip(-0.10, 0.10).copy()

    print(f"Clean dataset shape: {X.shape}")

    max_rows = 120000
    if len(X) > max_rows:
        X = X.tail(max_rows).copy()
        y_cls = y_cls.tail(max_rows).copy()
        y_reg = y_reg.tail(max_rows).copy()
        print(f"Trimmed dataset shape: {X.shape}")

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print("Training classifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train_cls, y_train_cls)

    print("Training regressor...")
    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train_reg, y_train_reg)

    print("Evaluating classifier...")
    y_pred_cls = clf.predict(X_test_cls)
    acc = accuracy_score(y_test_cls, y_pred_cls)
    print(f"Classifier Accuracy: {acc:.4f}")
    print(classification_report(y_test_cls, y_pred_cls))

    print("Evaluating regressor...")
    y_pred_reg = reg.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    print(f"Regressor MAE: {mae:.6f}")

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(reg, REG_MODEL_PATH)

    print(f"Saved classifier: {MODEL_PATH}")
    print(f"Saved regressor: {REG_MODEL_PATH}")
    print(f"Total training time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    train_model()