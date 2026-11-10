"""
Reproducibility Validation Module
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_and_predict():
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    return y_pred, y_prob, metrics

def validate_reproducibility():
    print("\n--- Medical Appointment Reproducibility Validation ---")

    # Run the same deterministic training process twice
    predictions_1, probabilities_1, metrics_1 = train_and_predict()
    predictions_2, probabilities_2, metrics_2 = train_and_predict()

    predictions_match = np.array_equal(predictions_1, predictions_2)
    probabilities_match = np.allclose(probabilities_1, probabilities_2)

    metrics_match = all(
        np.isclose(metrics_1[key], metrics_2[key])
        for key in metrics_1
    )

    print(f"Predictions identical : {predictions_match}")
    print(f"Probabilities identical: {probabilities_match}")
    print(f"Metrics identical     : {metrics_match}")

    if predictions_match and probabilities_match and metrics_match:
        print("[SUCCESS] Reproducibility validation PASSED!")
        print(
            f"Verified Metrics: Accuracy: {metrics_1['accuracy']:.4f} | "
            f"F1: {metrics_1['f1_score']:.4f} | "
            f"ROC-AUC: {metrics_1['roc_auc']:.4f}"
        )
        return True
    else:
        print("[ERROR] Reproducibility validation FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    validate_reproducibility()
