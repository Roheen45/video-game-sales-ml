# Video Game Sales ML demo

This project uses the included `vgsales.csv` dataset to demonstrate two machine-learning tasks. It predicts a game's `Global_Sales` from its platform, genre, and release year, and separately classifies the game as **Low Sales** or **High Sales**. Regional sales columns are deliberately excluded because they directly reveal the global total.

## Setup and run

Run these commands from this folder with Python installed:

1. (Optional) Create and activate a virtual environment.
2. Install packages: `python -m pip install -r requirements.txt`
3. Train both models and report held-out metrics: `python train_models.py`
4. Start the API and webpage: `uvicorn main:app --reload`
5. Open <http://127.0.0.1:8000>. Interactive API documentation is at <http://127.0.0.1:8000/docs>.

The dataset is already provided as `vgsales.csv`; training does not need to extract or replace it. The models are saved locally as `models/regression.joblib` and `models/classification.joblib`.

## Models and inputs

Both endpoints accept JSON with `platform`, `genre`, and `year`.

- `POST /predict/regression` estimates `Global_Sales` in millions of copies. A preprocessing pipeline imputes missing years with the training median and one-hot encodes platform and genre. A gradient boosting regressor predicts the continuous target.
- `POST /predict/classification` returns `Low Sales` when `Global_Sales < 1.0` million and `High Sales` when it is at least 1.0 million. `Global_Sales` is used only to create the target labels, never as an input. The classifier is a balanced random forest with the same preprocessing.

Both models use an 80/20 train/test split with random seed 42. Classification preserves the class proportions with stratification. Metrics below are from the held-out test set, not the training data.

## Evaluation results

Results from `python train_models.py` on the included 16,598-row dataset:

- Regression R²: **0.0173**
- Classification accuracy: **0.6569** (65.69%)

The regression result is weak: platform, genre, and year alone explain little of the variation in sales, especially for unusually popular games. Treat the prediction as a demonstration, not a reliable sales forecast. Classification accuracy is also a baseline result and should be interpreted with the stated 1.0-million threshold.
