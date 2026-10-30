"""Train and evaluate the regression and classification models for the lab."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "vgsales.csv"
MODEL_DIR = ROOT / "models"
SALES_FEATURES = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
GAME_FEATURES = ["Platform", "Year"]
CATEGORICAL = ["Platform"]


def game_preprocessor():
    return ColumnTransformer([
        ("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
        ]), CATEGORICAL),
        ("year", SimpleImputer(strategy="median"), ["Year"]),
    ])


def naive_bayes_preprocessor():
    return ColumnTransformer([
        ("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
        ]), CATEGORICAL),
        ("year", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("bins", KBinsDiscretizer(n_bins=8, encode="onehot", strategy="quantile")),
        ]), ["Year"]),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Place vgsales.csv beside this script and run it again.")
    df = pd.read_csv(DATA_PATH).dropna(subset=["Global_Sales"])
    metrics = {"regression": {}, "classification": {}}

    X = df[SALES_FEATURES].fillna(0)
    y = df["Global_Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regression_models = {
        "linear": (LinearRegression(), ["NA_Sales"]),
        "multiple": (LinearRegression(), SALES_FEATURES),
        "polynomial": (Pipeline([
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("regressor", LinearRegression()),
        ]), SALES_FEATURES),
        "lasso": (Pipeline([
            ("scale", StandardScaler()),
            ("regressor", Lasso(alpha=0.01, max_iter=10000)),
        ]), SALES_FEATURES),
    }
    for name, (model, features) in regression_models.items():
        model.fit(X_train[features], y_train)
        metrics["regression"][name] = round(float(r2_score(y_test, model.predict(X_test[features]))), 4)
        joblib.dump(model, MODEL_DIR / f"regression_{name}.joblib")

    games = df.dropna(subset=["Genre", "Platform", "Year"]).copy()
    Xg, yg = games[GAME_FEATURES], games["Genre"]
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        Xg, yg, test_size=0.2, random_state=42, stratify=yg,
    )
    classifier_specs = {
        "logistic": OneVsRestClassifier(LogisticRegression(max_iter=1200, solver="liblinear", class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(max_depth=18, class_weight="balanced", random_state=42),
        "naive_bayes": MultinomialNB(),
        "svm": LinearSVC(class_weight="balanced", random_state=42, max_iter=5000),
    }
    for name, estimator in classifier_specs.items():
        features = naive_bayes_preprocessor() if name == "naive_bayes" else game_preprocessor()
        model = Pipeline([("features", features), ("classifier", estimator)])
        model.fit(Xg_train, yg_train)
        metrics["classification"][name] = round(float(accuracy_score(yg_test, model.predict(Xg_test))), 4)
        joblib.dump(model, MODEL_DIR / f"classification_{name}.joblib")

    text_model = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=25000, min_df=2)),
        ("svm", LinearSVC(class_weight="balanced", random_state=42, max_iter=5000)),
    ])
    Xname = games["Name"].fillna("Unknown game").astype(str)
    name_train, name_test, genre_train, genre_test = train_test_split(
        Xname, yg, test_size=0.2, random_state=42, stratify=yg,
    )
    text_model.fit(name_train, genre_train)
    metrics["classification"]["tfidf"] = round(float(accuracy_score(genre_test, text_model.predict(name_test))), 4)
    joblib.dump(text_model, MODEL_DIR / "classification_tfidf.joblib")

    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Held-out test scores (80/20 split, random_state=42):")
    for family, scores in metrics.items():
        score_name = "R²" if family == "regression" else "accuracy"
        for model_name, score in scores.items():
            print(f"{family}/{model_name}: {score_name}={score:.4f}")
    print(f"Saved model files and metrics in {MODEL_DIR}")


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    main()
