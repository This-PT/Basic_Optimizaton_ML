from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
N_FOLDS = 5
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def prediction_metrics(y_true, predictions):
    """Return the metrics needed for this imbalanced problem."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "yes_precision": precision_score(
            y_true,
            predictions,
            pos_label="yes",
            zero_division=0,
        ),
        "yes_recall": recall_score(
            y_true,
            predictions,
            pos_label="yes",
            zero_division=0,
        ),
        "yes_f1": f1_score(
            y_true,
            predictions,
            pos_label="yes",
            zero_division=0,
        ),
        "predicted_yes": int(np.sum(predictions == "yes")),
    }


def yes_probabilities(model, X_data):
    """Return the probability assigned to the positive class, 'yes'."""
    class_names = model.named_steps["classifier"].classes_
    yes_index = np.where(class_names == "yes")[0][0]
    return model.predict_proba(X_data)[:, yes_index]


def threshold_diagnostics(model_name, y_true, yes_scores):
    """Measure one model at every candidate threshold."""
    rows = []

    for threshold in THRESHOLDS:
        predictions = np.where(yes_scores >= threshold, "yes", "no")
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                **prediction_metrics(y_true, predictions),
            }
        )

    return pd.DataFrame(rows)


def fold_f1_scores(y_true, yes_scores, threshold, cv_splits):
    """Show how stable one selected threshold is across CV folds."""
    y_array = np.asarray(y_true)
    scores = []

    for _, validation_indices in cv_splits:
        fold_predictions = np.where(
            yes_scores[validation_indices] >= threshold,
            "yes",
            "no",
        )
        scores.append(
            f1_score(
                y_array[validation_indices],
                fold_predictions,
                pos_label="yes",
                zero_division=0,
            )
        )

    return np.asarray(scores)


# ============================================================
# 1. Load data and keep only features available before calling
# ============================================================

df = pd.read_csv("bank-full.csv", sep=";")
y = df["y"]

X = df.drop(
    columns=[
        "y",
        "duration",
        "contact",
        "day",
        "month",
        "campaign",
    ]
)


# ============================================================
# 2. Reserve the test set before model selection
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# 3. Build preprocessing inside the pipeline
# ============================================================

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


# ============================================================
# 4. Define candidate models
# ============================================================

models = {
    "logistic_regression": Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "decision_tree": Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=50,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}


# ============================================================
# 5. Create the same five stratified folds for every candidate
# ============================================================

cv = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)
cv_splits = list(cv.split(X_train, y_train))


# ============================================================
# 6. Produce out-of-fold probabilities and tune thresholds
# ============================================================

class_names = np.sort(y_train.unique())
yes_index = np.where(class_names == "yes")[0][0]

diagnostic_tables = []
best_settings = {}

for model_name, model in models.items():
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv_splits,
        method="predict_proba",
    )
    oof_yes_scores = oof_probabilities[:, yes_index]

    diagnostics = threshold_diagnostics(
        model_name,
        y_train,
        oof_yes_scores,
    )
    diagnostic_tables.append(diagnostics)

    best_row = diagnostics.loc[diagnostics["yes_f1"].idxmax()]
    best_threshold = float(best_row["threshold"])
    fold_scores = fold_f1_scores(
        y_train,
        oof_yes_scores,
        best_threshold,
        cv_splits,
    )

    best_settings[model_name] = {
        "threshold": best_threshold,
        "oof_f1": float(best_row["yes_f1"]),
        "fold_f1_mean": float(fold_scores.mean()),
        "fold_f1_std": float(fold_scores.std()),
    }

threshold_table = pd.concat(diagnostic_tables, ignore_index=True)


# ============================================================
# 7. Compare the dummy baseline with each candidate's best row
# ============================================================

dummy_model = DummyClassifier(strategy="most_frequent")
dummy_predictions = cross_val_predict(
    dummy_model,
    X_train,
    y_train,
    cv=cv_splits,
    method="predict",
)
dummy_metrics = prediction_metrics(y_train, dummy_predictions)

comparison_rows = [
    {
        "model": "dummy",
        "threshold": np.nan,
        **dummy_metrics,
        "fold_f1_mean": 0.0,
        "fold_f1_std": 0.0,
    }
]

for model_name, setting in best_settings.items():
    best_row = threshold_table[
        (threshold_table["model"] == model_name)
        & (threshold_table["threshold"] == setting["threshold"])
    ].iloc[0].to_dict()

    best_row["fold_f1_mean"] = setting["fold_f1_mean"]
    best_row["fold_f1_std"] = setting["fold_f1_std"]
    comparison_rows.append(best_row)

comparison_table = pd.DataFrame(comparison_rows)

selected_model_name = max(
    best_settings,
    key=lambda name: best_settings[name]["oof_f1"],
)
selected_threshold = best_settings[selected_model_name]["threshold"]


# ============================================================
# 8. Print only the model-selection diagnostics
# ============================================================

print("\nDATA")
print(
    f"rows={len(df)}  features={X.shape[1]}  "
    f"cv_train={len(X_train)}  test={len(X_test)}  "
    f"folds={N_FOLDS}  yes_rate={y.eq('yes').mean():.3f}"
)

print("\nOUT-OF-FOLD THRESHOLDS")
print(
    threshold_table[
        [
            "model",
            "threshold",
            "balanced_accuracy",
            "yes_precision",
            "yes_recall",
            "yes_f1",
            "predicted_yes",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
)

print("\nCROSS-VALIDATION MODEL COMPARISON")
print(
    comparison_table[
        [
            "model",
            "threshold",
            "balanced_accuracy",
            "yes_precision",
            "yes_recall",
            "yes_f1",
            "fold_f1_mean",
            "fold_f1_std",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
)

print(
    f"\nSELECTED model={selected_model_name} "
    f"threshold={selected_threshold:.2f}"
)


# ============================================================
# 9. Fit the selected configuration on all X_train
# ============================================================

final_model = clone(models[selected_model_name])
final_model.fit(X_train, y_train)


# ============================================================
# 10. Apply the selected threshold once to the test set
# ============================================================

test_yes_scores = yes_probabilities(final_model, X_test)
test_predictions = np.where(
    test_yes_scores >= selected_threshold,
    "yes",
    "no",
)
test_metrics = prediction_metrics(y_test, test_predictions)

print("\nFINAL TEST METRICS")
for metric_name, metric_value in test_metrics.items():
    if metric_name == "predicted_yes":
        print(f"{metric_name}={metric_value}")
    else:
        print(f"{metric_name}={metric_value:.3f}")

test_confusion = pd.DataFrame(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=["no", "yes"],
    ),
    index=["actual_no", "actual_yes"],
    columns=["predicted_no", "predicted_yes"],
)

print("\nFINAL TEST CONFUSION MATRIX")
print(test_confusion)


# ============================================================
# 11. Print one compact error diagnostic
# ============================================================

error_analysis = X_test.copy()
error_analysis["Actual"] = y_test.to_numpy()
error_analysis["Predicted"] = test_predictions
error_analysis["IsFalsePositive"] = (
    (error_analysis["Actual"] == "no")
    & (error_analysis["Predicted"] == "yes")
)
error_analysis["IsFalseNegative"] = (
    (error_analysis["Actual"] == "yes")
    & (error_analysis["Predicted"] == "no")
)

poutcome_diagnostic = (
    error_analysis.groupby("poutcome")
    .agg(
        Customers=("Actual", "size"),
        ActualYes=("Actual", lambda values: (values == "yes").sum()),
        PredictedYes=("Predicted", lambda values: (values == "yes").sum()),
        FalsePositives=("IsFalsePositive", "sum"),
        FalseNegatives=("IsFalseNegative", "sum"),
    )
    .sort_index()
)

print("\nERRORS BY PREVIOUS CAMPAIGN OUTCOME")
print(poutcome_diagnostic)


# ============================================================
# 12. Explain which features the selected tree uses most
# ============================================================

if selected_model_name == "decision_tree":
    feature_names = final_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_values = final_model.named_steps[
        "classifier"
    ].feature_importances_

    feature_importance = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance_values,
            }
        )
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    print("\nTOP 10 IMPORTANT FEATURES")
    print(
        feature_importance.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

# ============================================================
# 13. Save the trained pipeline and its decision settings
# ============================================================

artifact_directory = Path("artifacts")
artifact_directory.mkdir(exist_ok=True)
artifact_path = artifact_directory / "bank_marketing_model.joblib"

model_bundle = {
    "model": final_model,
    "threshold": selected_threshold,
    "model_name": selected_model_name,
    "feature_names": X.columns.tolist(),
}

joblib.dump(model_bundle, artifact_path)
print(f"\nMODEL SAVED TO\n{artifact_path.resolve()}")

