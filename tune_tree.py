"""Compare a small tree-setting grid with five-fold out-of-fold predictions.

This script uses only the development split. It does not evaluate X_test.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
N_FOLDS = 5
DEPTHS = [4, 6, 8]
MIN_LEAF_SIZES = [50, 100]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


df = pd.read_csv("bank-full.csv", sep=";")
y = df["y"]
X = df.drop(
    columns=["y", "duration", "contact", "day", "month", "campaign"]
)

# Recreate the same development/test split as baseline.py.
# The test rows are deliberately not used in this experiment.
X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()


def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


cv = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)
cv_splits = list(cv.split(X_train, y_train))
class_names = np.sort(y_train.unique())
yes_index = np.where(class_names == "yes")[0][0]
y_array = y_train.to_numpy()

rows = []

for depth in DEPTHS:
    for min_leaf in MIN_LEAF_SIZES:
        candidate = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_leaf=min_leaf,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        # Every row is predicted by a tree that did not train on that row.
        oof_probabilities = cross_val_predict(
            candidate,
            X_train,
            y_train,
            cv=cv_splits,
            method="predict_proba",
        )[:, yes_index]

        for threshold in THRESHOLDS:
            predictions = np.where(
                oof_probabilities >= threshold,
                "yes",
                "no",
            )

            fold_f1 = []
            for _, validation_indices in cv_splits:
                fold_f1.append(
                    f1_score(
                        y_array[validation_indices],
                        predictions[validation_indices],
                        pos_label="yes",
                        zero_division=0,
                    )
                )

            rows.append(
                {
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "threshold": threshold,
                    "precision": precision_score(
                        y_train,
                        predictions,
                        pos_label="yes",
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        y_train,
                        predictions,
                        pos_label="yes",
                        zero_division=0,
                    ),
                    "oof_f1": f1_score(
                        y_train,
                        predictions,
                        pos_label="yes",
                        zero_division=0,
                    ),
                    "fold_f1_std": np.std(fold_f1),
                }
            )

results = pd.DataFrame(rows)

# Keep one best threshold per tree configuration. If F1 ties exactly,
# the lower threshold appears first because THRESHOLDS is ascending.
best_per_tree = (
    results.sort_values(
        ["oof_f1", "threshold"],
        ascending=[False, True],
    )
    .drop_duplicates(["max_depth", "min_samples_leaf"])
    .sort_values(["max_depth", "min_samples_leaf"])
)

print("\nBEST TESTED THRESHOLD FOR EACH TREE SETTING")
print(
    best_per_tree.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

winner = results.loc[results["oof_f1"].idxmax()]
print("\nHIGHEST OOF F1 AMONG TESTED COMBINATIONS")
print(
    f"max_depth={int(winner['max_depth'])} "
    f"min_samples_leaf={int(winner['min_samples_leaf'])} "
    f"threshold={winner['threshold']:.2f} "
    f"oof_f1={winner['oof_f1']:.3f}"
)
