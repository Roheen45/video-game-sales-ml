"""Train sales regression and sales-level classification models."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "vgsales.csv"
MODELS = ROOT / "models"
SALES_THRESHOLD = 1.0
RANDOM_STATE = 42
FEATURES = ["Platform", "Genre", "Year"]
CATEGORICAL_FEATURES = ["Platform", "Genre"]
NUMERIC_FEATURES = ["Year"]


def make_preprocessor():
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numeric", numeric, NUMERIC_FEATURES),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Put vgsales.csv beside this script and run it again.")

    df = pd.read_csv(DATA_PATH).dropna(subset=["Global_Sales"])
    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, df["Global_Sales"], test_size=0.2, random_state=RANDOM_STATE
    )
    regression = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", GradientBoostingRegressor(
            n_estimators=200, max_depth=2, random_state=RANDOM_STATE,
        )),
    ])
    regression.fit(X_train, y_train)
    regression_r2 = r2_score(y_test, regression.predict(X_test))

    # The target defines labels only. Global_Sales is never part of X/features.
    labels = (df["Global_Sales"] >= SALES_THRESHOLD).map({True: "High Sales", False: "Low Sales"})
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    classification = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1,
        )),
    ])
    classification.fit(X_train_c, y_train_c)
    classification_accuracy = accuracy_score(y_test_c, classification.predict(X_test_c))

    MODELS.mkdir(exist_ok=True)
    joblib.dump(regression, MODELS / "regression.joblib")
    joblib.dump(classification, MODELS / "classification.joblib")
    print(f"Sales threshold: {SALES_THRESHOLD:.1f} million (High Sales is >= threshold)")
    print(f"Regression held-out test R2: {regression_r2:.4f}")
    print(f"Classification held-out test accuracy: {classification_accuracy:.4f}")
    print(f"Saved models in {MODELS}")


if __name__ == "__main__":
    main()
