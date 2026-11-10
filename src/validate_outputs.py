"""
Output Validation Module
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import json
import numpy as np

def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs for Medical Appointment No-Shows...")
    
    # 1. Load pipeline outputs
    try:
        X_train = np.load('data/processed/X_train_final.npy')
        X_test = np.load('data/processed/X_test_final.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find processed data: {e}")
        return False
        
    errors = []
    
    # 2. Check for Missing Values (NaNs)
    nan_train = int(np.isnan(X_train).sum())
    nan_test = int(np.isnan(X_test).sum())
    nan_y_train = int(np.isnan(y_train).sum())
    nan_y_test = int(np.isnan(y_test).sum())
    
    if nan_train > 0:
        errors.append(f"NaNs detected in X_train: {nan_train}")
    if nan_test > 0:
        errors.append(f"NaNs detected in X_test: {nan_test}")
    if nan_y_train > 0:
        errors.append("NaNs detected in y_train")
    if nan_y_test > 0:
        errors.append("NaNs detected in y_test")
        
    # 3. Check Dimensionality and Alignment
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature count mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append(f"Row mismatch between X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}).")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append(f"Row mismatch between X_test ({X_test.shape[0]}) and y_test ({y_test.shape[0]}).")
        
    # 4. Check Target Values
    unique_y_train = np.unique(y_train).tolist()
    unique_y_test = np.unique(y_test).tolist()
    if set(unique_y_train) - {0, 1}:
        errors.append(f"Invalid target values in y_train: {unique_y_train}")
    if set(unique_y_test) - {0, 1}:
        errors.append(f"Invalid target values in y_test: {unique_y_test}")
        
    # 5. Generate Preprocessing Summary Report
    report = {
        "project": "Medical Appointment No-Show Prediction",
        "student_registration": "241FA18343",
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape)
        },
        "target_balance": {
            "train_show_0": int((y_train == 0).sum()),
            "train_no_show_1": int((y_train == 1).sum()),
            "train_no_show_rate_percent": round(float((y_train == 1).mean() * 100), 2),
            "test_show_0": int((y_test == 0).sum()),
            "test_no_show_1": int((y_test == 1).sum()),
            "test_no_show_rate_percent": round(float((y_test == 1).mean() * 100), 2)
        },
        "data_quality": {
            "missing_values_X_train": nan_train,
            "missing_values_X_test": nan_test,
            "missing_values_y_train": nan_y_train,
            "missing_values_y_test": nan_y_test
        },
        "errors": errors
    }
    
    os.makedirs("artifacts", exist_ok=True)
    report_path = 'artifacts/preprocessing_summary_report.json'
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    if errors:
        print("[ERROR] Output Validation FAILED!")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("[SUCCESS] Output Validation PASSED.")
        print(f"[INFO] Features verified: {X_train.shape[1]} features, {X_train.shape[0]} train rows, {X_test.shape[0]} test rows.")
        print(f"[INFO] Summary report saved to {report_path}")
        return True

if __name__ == "__main__":
    validate_preprocessing_outputs()
