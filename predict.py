from pathlib import Path

import joblib
import pandas as pd


ARTIFACT_PATH = Path("artifacts") / "bank_marketing_model.joblib"


def load_model_bundle(artifact_path=ARTIFACT_PATH):
    """Load the trained pipeline, threshold, and model metadata."""
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"{artifact_path} does not exist. Run python baseline.py first."
        )

    return joblib.load(artifact_path)


def predict_customer(customer, model_bundle):
    """Return the yes probability and final label for one customer."""
    required_features = model_bundle["feature_names"]
    missing_features = [
        feature for feature in required_features if feature not in customer
    ]

    if missing_features:
        raise ValueError(
            f"Missing customer features: {', '.join(missing_features)}"
        )

    customer_frame = pd.DataFrame(
        [{feature: customer[feature] for feature in required_features}]
    )

    model = model_bundle["model"]
    threshold = model_bundle["threshold"]
    class_names = model.named_steps["classifier"].classes_
    yes_index = list(class_names).index("yes")
    yes_probability = float(
        model.predict_proba(customer_frame)[0, yes_index]
    )
    prediction = "yes" if yes_probability >= threshold else "no"

    return {
        "prediction": prediction,
        "yes_probability": yes_probability,
        "threshold": threshold,
        "model_name": model_bundle["model_name"],
    }


def main():
    model_bundle = load_model_bundle()

    sample_customer = {
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "no",
        "loan": "no",
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }

    result = predict_customer(sample_customer, model_bundle)

    print("SAMPLE CUSTOMER PREDICTION")
    print(f"model={result['model_name']}")
    print(f"yes_probability={result['yes_probability']:.3f}")
    print(f"threshold={result['threshold']:.2f}")
    print(f"prediction={result['prediction']}")


if __name__ == "__main__":
    main()

