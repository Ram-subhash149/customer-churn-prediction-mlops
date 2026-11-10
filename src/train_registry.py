"""
Model Registry Training & Registration Module
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import numpy as np
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry for Medical Appointments ---")
    
    # 1. Set up MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_name = "No_Show_Production_Model"
    mlflow.set_experiment("Medical_Appointment_No_Show_Prediction")
    
    # 2. Load Processed Data
    try:
        X_train = np.load('data/processed/X_train_final.npy')
        X_test = np.load('data/processed/X_test_final.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run preprocessing first.")
        return
        
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count)
    
    # 3. Define Candidate Parameters for Production Registry
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1
    }
    
    with mlflow.start_run(run_name="XGBoost_Registry_Production_Candidate") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "XGBoost")
        
        # 4. Train Model
        print(f"[INFO] Training XGBoost candidate for {model_name}...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        print("[INFO] Evaluating candidate model...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        mlflow.log_metrics(metrics)
        
        # 6. Log Preprocessor Pipeline for Lineage
        if os.path.exists("models/preprocessor.pkl"):
            print("[INFO] Attaching preprocessing pipeline to model artifacts...")
            mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")
            
        # 7. Log and Register the Model
        print(f"[INFO] Pushing model to MLflow Registry under '{model_name}'...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=model_name
        )
        
        print(f"[SUCCESS] Metrics logged: ROC-AUC = {metrics['roc_auc']:.4f} | F1 = {metrics['f1_score']:.4f} | Recall = {metrics['recall']:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: '{model_name}'")

if __name__ == "__main__":
    train_and_register_model()
