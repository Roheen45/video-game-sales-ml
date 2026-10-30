# Arcade Lab — ML model explorer

A small FastAPI and HTML app for comparing regression and classification models on the included Video Game Sales dataset.

## Run it

From this folder, run:

```powershell
python -m pip install -r requirements.txt
python train_models.py
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/docs>.

## Included techniques

- **Regression:** simple linear regression, multiple linear regression, polynomial regression (degree 2), and Lasso.
- **Classification:** logistic regression, random forest, decision tree, Naive Bayes, and linear SVM. These predict game genre from platform and year.
- **Text classification:** TF-IDF with a linear SVM predicts genre from the game title.

Choose a regression or classification model from its dropdown. Regression uses the four regional sales values. Classification uses the selected platform and year; TF-IDF uses the entered game title. Held-out test scores are shown below the forms and saved to `models/metrics.json`. All fitted estimators are saved as joblib files in `models/`.

## Dataset and evaluation

The supplied `vgsales.csv` has 16,598 games from 1980 through 2020. Models use an 80/20 train/test split with random seed 42. Game sales figures in the regression task are measured in millions of copies. Global sales is the sum of the four regional sales fields, so using all four makes a strong, largely arithmetic baseline; treat it as a regression demonstration rather than an unknown-sales forecast. PS5 and Switch are available in the platform menu but have no examples in this dataset, so predictions for them are less reliable. Genre is imbalanced, so classification accuracy is a simple comparison metric and does not describe every genre equally well.

See [REPORT.md](REPORT.md) for the short project report and the measured scores.
