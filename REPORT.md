# Lab Midterm Project — Brief Report

## Objective

Apply the regression, classification, and text-feature techniques covered in the lab to a suitable dataset. The project uses the provided Video Game Sales CSV and exposes predictions through FastAPI with a connected browser interface.

## Data and preparation

The dataset contains 16,598 game records from 1980 through 2020. Missing regional sales values are filled with zero for regression. The genre classifiers use platform and release year; missing values are imputed inside each scikit-learn pipeline and platform is one-hot encoded. The text model uses game names and a fitted TF-IDF vectorizer. A fixed random seed and 80/20 train/test split make the reported results repeatable. PS5 and Switch do not appear in the training data, so predictions for those menu options are less reliable.

## Models and held-out results

| Task | Technique | Test result |
|---|---|---:|
| Regression | Linear regression (North America sales only) | R² = 0.9247 |
| Regression | Multiple linear regression (four regional sales) | R² = 1.0000 |
| Regression | Polynomial regression, degree 2 | R² = 1.0000 |
| Regression | Lasso regression | R² = 0.9999 |
| Classification | One-vs-rest logistic regression: genre from platform and year | Accuracy = 0.1834 |
| Classification | Random forest: genre from platform and year | Accuracy = 0.1620 |
| Classification | Decision tree: genre from platform and year | Accuracy = 0.1632 |
| Classification | Multinomial Naive Bayes with binned release year: genre from platform and year | Accuracy = 0.2208 |
| Classification | Linear SVM: genre from platform and year | Accuracy = 0.1445 |
| Text classification | TF-IDF + linear SVM: genre from title | Accuracy = 0.7492 |

## Interpretation

The four regional sales figures nearly determine `Global_Sales` by definition, which explains the very high regression R² values. They are useful for demonstrating the algorithms but do not represent a realistic forecast from information known before sales occur. Platform and release year alone are weak signals for genre in this dataset; the title-based TF-IDF model performs better because words in game names often indicate genre. The dataset is genre-imbalanced, so accuracy should be interpreted as an overall comparison rather than equal performance across all genres.

## Application

The FastAPI backend loads joblib estimators and provides `/predict/regression` and `/predict/classification`; `/metrics` serves the held-out scores. The HTML/CSS/JavaScript page has a simple form for each task, model and platform dropdowns, a prediction output, and score lists. Setup and launch commands are in `README.md`.
