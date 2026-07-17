# Titanic: Machine Learning from Disaster

A binary classification project for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) — predicting passenger survival from demographic and travel data.

## Overview

Using features such as age, sex, passenger class, fare and family composition, the goal is to predict whether each passenger survived (1) or did not (0).

## Approach

1. **Exploratory Data Analysis** — using [QuickEDA](https://github.com/wdeltenre/QuickEDA) to sanity-check the raw data and intermediate transformations.
2. **Preprocessing** — train/validation/test split, missing value imputation, categorical encoding, feature scaling.
3. **Feature Engineering** — family size, "is alone" indicator.
4. **Model Training** — Logistic Regression, KNN, Random Forest and Gradient Boosting, tuned with `GridSearchCV`.
5. **Evaluation** — compared on a held-out test set using accuracy, precision, recall and F1.

## Results

| Model | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|---|---|---|---|---|
| XGBoost | 0.7933 | 0.82 | 0.59 | 0.69 |
| Logistic Regression | 0.7877 | 0.78 | 0.62 | 0.69 |
| Random Forest | 0.7877 | 0.79 | 0.61 | 0.69 |
| Gradient Boosting | 0.7765 | 0.81 | 0.55 | 0.66 |
| KNN | 0.7654 | 0.71 | 0.67 | 0.69 |

**XGBoost** edged out the field at **79.3% accuracy**, though the margin over Logistic Regression and Random Forest is minimal — suggesting survival patterns are largely linearly separable given the engineered features. All models struggle most with recall on survivors (class 1), staying conservative toward the majority "did not survive" class.

Key survival factors such as exact location on the ship at impact and proximity to lifeboats aren't in the data, which caps how far any model can improve.

## Tech stack

- Python 3.11, pandas, NumPy
- scikit-learn (Logistic Regression, KNN, Random Forest, GridSearchCV)
- XGBoost (Gradient Boosting variant)
- matplotlib, seaborn (visualization)

## Project structure

```
titanic/
├── titanic.ipynb            # full pipeline: EDA → preprocessing → models → evaluation
├── Titanic-Dataset.csv       # Kaggle training data
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/wdeltenre/titanic.git
cd titanic
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Acknowledgments

EDA powered by [QuickEDA](https://github.com/wdeltenre/QuickEDA), a Streamlit tool I designed and built myself to speed up dataset exploration across all my projects.
