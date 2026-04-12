import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split

from config import FEATURE_COLUMNS, MERGED_DATASET_PATH, MODEL_PATH, REG_MODEL_PATH


def train_model():
    start_time = time.time()

    print("Loading merged dataset...")
    df = pd.read_csv(MERGED_DATASET_PATH)

    # 清掉欄位前後空白，避免明明有欄位但抓不到
    df.columns = df.columns.str.strip()

    print(f"Dataset shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())

    # 檢查必要特徵欄位
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    # 如果沒有 Target_Return，就自動建立
    if "Target_Return" not in df.columns:
        if "Close" not in df.columns:
            raise ValueError(
                "Target_Return not found, and Close column is also missing. "
                "Cannot create Target_Return automatically."
            )
        print("Target_Return not found. Creating it from Close...")
        df["Target_Return"] = df["Close"].pct_change().shift(-1)

    # 如果沒有 Target，就根據 Target_Return 建立分類
    if "Target" not in df.columns:
        print("Target not found. Creating it from Target_Return...")
        df["Target"] = (df["Target_Return"] > 0).astype(int)

    X = df[FEATURE_COLUMNS].copy()
    y_cls = df["Target"].copy()
    y_reg = df["Target_Return"].copy()

    # 把 inf 變成 NA
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    y_reg = y_reg.replace([float("inf"), float("-inf")], pd.NA)

    # 合併後一起 dropna，避免長度不一致
    merged = pd.concat([X, y_cls, y_reg], axis=1).dropna()

    X = merged[FEATURE_COLUMNS]
    y_cls = merged["Target"]
    y_reg = merged["Target_Return"]

    print(f"Clean dataset shape: {X.shape}")

    max_rows = 120000
    if len(X) > max_rows:
        X = X.tail(max_rows).copy()
        y_cls = y_cls.tail(max_rows).copy()
        y_reg = y_reg.tail(max_rows).copy()
        print(f"Trimmed dataset to last {max_rows} rows")

    print("Splitting train/test...")
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, shuffle=False
    )

    print("Training classifier...")
    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    clf.fit(X_train, y_cls_train)

    print("Training regressor...")
    reg = RandomForestRegressor(
        n_estimators=80,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    reg.fit(X_train, y_reg_train)

    print("Evaluating classifier...")
    pred_cls = clf.predict(X_test)
    acc = accuracy_score(y_cls_test, pred_cls)

    print("\n=== Classification Result ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_cls_test, pred_cls))

    print("Evaluating regressor...")
    pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, pred_reg)

    print("\n=== Regression Result ===")
    print(f"MAE (next-day return): {mae:.6f}")

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(reg, REG_MODEL_PATH)

    print(f"Saved classifier to: {MODEL_PATH}")
    print(f"Saved regressor to: {REG_MODEL_PATH}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.2f} seconds")

    return clf, reg


if __name__ == "__main__":
    train_model()