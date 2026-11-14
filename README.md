# Machine Learning Models

Hands-on experiments with classical machine learning: building preprocessing pipelines, comparing models, and tuning hyperparameters. Every result here comes from my own runs.

## Project 1: Titanic Survival Prediction

**Goal:** predict which passengers survived the Titanic disaster ([Kaggle competition](https://www.kaggle.com/competitions/titanic)), using the passenger data: class, sex, age, fare, family aboard and port of embarkation.

### What I built

- **A scikit-learn `Pipeline` + `ColumnTransformer`.** Numeric columns get median imputation and standard scaling. Categorical columns get most-frequent imputation and one-hot encoding. Everything is fitted inside the pipeline, so no preprocessing leaks from the validation set.
- **A model comparison.** I compared Random Forest, XGBoost and a Decision Tree on the same split.
- **`smart_tune()`, a custom adaptive hyperparameter search.** Instead of one large grid, it runs a small 2-value grid per parameter, finds the best value, then *shifts the grid toward it* and repeats. It's a hand-rolled hill-climbing search built on top of `RandomizedSearchCV`/`GridSearchCV`.
- **Overfitting checks.** I log train accuracy, validation accuracy and 5-fold CV mean ± std for every configuration.

### Results (80/20 hold-out split)

| Model | Validation accuracy | 5-fold CV (train) |
|---|---|---|
| Decision Tree (baseline) | 0.788 | — |
| XGBoost (default) | 0.782 | — |
| Random Forest (default) | 0.827 | — |
| **Random Forest + `smart_tune`** | **0.855–0.860** | **0.81 ± 0.015** |

`smart_tune` converged on `n_estimators ≈ 220–250`, `max_depth = 12–13`, `min_samples_split = 3–4` and `min_samples_leaf = 1`. The train/validation gap went from 1.00 / 0.86 with hand-picked parameters to 0.88 / 0.86, which means much less overfitting.

### Honest limitations and next steps

- The train/validation split has no fixed `random_state`, so the exact numbers change between runs. The CV score is the more reliable figure.
- `smart_tune` can step `min_samples_leaf` down to 0, which is invalid, so some fits fail. It needs a lower bound of 1.
- `PassengerId` is used as a feature (it's just a row number), and `Ticket`/`Cabin` are one-hot encoded raw. Next step: drop the ID and engineer `Title`, `FamilySize` and `Deck`.
- I haven't made a Kaggle leaderboard submission yet.

## Files

| File | What it is |
|---|---|
| `machine_Learning.ipynb` | The main Titanic pipeline and `smart_tune` ([open in Colab](https://colab.research.google.com/github/tas-tasnim/Machine-Learning-Models/blob/main/machine_Learning.ipynb)) |
| `Titanic Data set model/machine-l-2.ipynb` | An earlier exploration: encoders, model comparison, tree-count sweep |

## Run it

The notebooks were built on Kaggle/Colab. To run one, open it in Colab, sign in with `kagglehub.login()`, and the first cells download the dataset.

**Stack:** Python · pandas · NumPy · scikit-learn · XGBoost
