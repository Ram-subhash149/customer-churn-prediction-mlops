"""
Model Evaluation and Error Analysis Module
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_evaluation():
    print("[INFO] Starting Model Evaluation for Medical Appointment No-Shows...")
    
    # 1. Load test data and trained model
    X_test_final = np.load('data/processed/X_test_final.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    model_path = 'models/random_forest_baseline.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model = joblib.load(model_path)
    
    # 2. Make predictions and probabilities
    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # 3. Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "=" * 45)
    print("--- Medical Appointment Model Evaluation Report ---")
    print("=" * 45)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("=" * 45 + "\n")
    
    # 4. Error Analysis Export
    # Reconstruct test dataframe matching the exact split order
    raw_df = pd.read_csv('data/raw/KaggleV2-May-2016.csv')
    raw_df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
    raw_df = raw_df[raw_df['Age'] >= 0].copy()
    
    raw_df['ScheduledDay_dt'] = pd.to_datetime(raw_df['ScheduledDay'], utc=True)
    raw_df['AppointmentDay_dt'] = pd.to_datetime(raw_df['AppointmentDay'], utc=True)
    raw_df['WaitingDays'] = (
        raw_df['AppointmentDay_dt'].dt.normalize() - raw_df['ScheduledDay_dt'].dt.normalize()
    ).dt.days
    raw_df = raw_df[raw_df['WaitingDays'] >= 0].copy()
    raw_df['Target'] = (raw_df['No-show'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    
    # Same chronological sort before split
    raw_df = raw_df.sort_values(['PatientId', 'AppointmentDay_dt', 'ScheduledDay_dt']).reset_index(drop=True)
    
    _, X_test_raw = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df['Target']
    )
    
    analysis_df = X_test_raw.copy().reset_index(drop=True)
    analysis_df['Actual_NoShow'] = y_test
    analysis_df['Predicted_NoShow'] = y_pred
    analysis_df['Predicted_Probability'] = np.round(y_prob, 4)
    
    # Identify False Negatives and False Positives
    # False Negatives: Patient did NOT show up (Actual=1), but model predicted they WOULD show up (Predicted=0)
    false_negatives = analysis_df[(analysis_df['Actual_NoShow'] == 1) & (analysis_df['Predicted_NoShow'] == 0)]
    # False Positives: Patient showed up (Actual=0), but model predicted they would NOT show up (Predicted=1)
    false_positives = analysis_df[(analysis_df['Actual_NoShow'] == 0) & (analysis_df['Predicted_NoShow'] == 1)]
    
    os.makedirs('outputs', exist_ok=True)
    analysis_df.to_csv('outputs/predictions.csv', index=False)
    false_negatives.to_csv('outputs/false_negatives.csv', index=False)
    false_positives.to_csv('outputs/false_positives.csv', index=False)
    
    print(f"[INFO] Exported predictions: {len(analysis_df)} records -> outputs/predictions.csv")
    print(f"[INFO] Exported False Negatives: {len(false_negatives)} records -> outputs/false_negatives.csv")
    print(f"[INFO] Exported False Positives: {len(false_positives)} records -> outputs/false_positives.csv")
    print("[SUCCESS] Evaluation and Error Analysis completed successfully!")

if __name__ == "__main__":
    run_evaluation()
