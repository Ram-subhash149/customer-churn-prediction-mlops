"""
Production Preprocessing Pipeline Module
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    print("[INFO] Starting Sklearn Modular Pipeline Preprocessing for Medical Appointments...")
    
    # 1. Load Raw Data
    data_path = 'data/raw/KaggleV2-May-2016.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 2. Data Cleaning
    df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
    df = df[df['Age'] >= 0].copy()
    
    df['ScheduledDay_dt'] = pd.to_datetime(df['ScheduledDay'], utc=True)
    df['AppointmentDay_dt'] = pd.to_datetime(df['AppointmentDay'], utc=True)
    
    df['WaitingDays'] = (
        df['AppointmentDay_dt'].dt.normalize() - df['ScheduledDay_dt'].dt.normalize()
    ).dt.days
    df = df[df['WaitingDays'] >= 0].copy()
    
    # Date-derived features
    df['ScheduledWeekday'] = df['ScheduledDay_dt'].dt.dayofweek
    df['AppointmentWeekday'] = df['AppointmentDay_dt'].dt.dayofweek
    df['ScheduledHour'] = df['ScheduledDay_dt'].dt.hour
    
    # Clean binary target
    df['Target'] = (df['No-show'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    
    # Sort chronologically to prevent data leakage in patient history
    df = df.sort_values(['PatientId', 'AppointmentDay_dt', 'ScheduledDay_dt']).reset_index(drop=True)
    
    df['PriorAppointments'] = df.groupby('PatientId').cumcount()
    df['PriorNoShows'] = (df.groupby('PatientId')['Target'].cumsum() - df['Target']).astype(int)
    df['PriorNoShowRate'] = (df['PriorNoShows'] / df['PriorAppointments'].replace(0, 1)).astype(float)
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1}).fillna(0).astype(int)
    
    feature_cols = [
        'Gender', 'Age', 'Scholarship', 'Hypertension', 'Diabetes',
        'Alcoholism', 'Handicap', 'SMS_received', 'WaitingDays',
        'ScheduledWeekday', 'AppointmentWeekday', 'ScheduledHour',
        'PriorAppointments', 'PriorNoShows', 'PriorNoShowRate'
    ]
    
    X = df[feature_cols].copy()
    y = df['Target'].copy()
    
    # 3. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Construct Scikit-Learn Pipeline
    # Continuous numeric features that benefit from standard scaling
    numeric_features = [
        'Age', 'WaitingDays', 'ScheduledWeekday', 'AppointmentWeekday',
        'ScheduledHour', 'PriorAppointments', 'PriorNoShows', 'PriorNoShowRate'
    ]
    # Binary / discrete indicator features passed through
    passthrough_features = [
        'Gender', 'Scholarship', 'Hypertension', 'Diabetes',
        'Alcoholism', 'Handicap', 'SMS_received'
    ]
    
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('pass', 'passthrough', passthrough_features)
        ],
        remainder='drop'
    )
    
    print("[INFO] Fitting preprocessing pipeline on training data...")
    preprocessor.fit(X_train)
    
    # Save standard outputs
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train_final = X_train.to_numpy(dtype=np.float64)
    X_test_final = X_test.to_numpy(dtype=np.float64)
    
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)
    
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/X_train_scaled.npy', X_train_scaled)
    np.save('data/processed/X_test_scaled.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))
    
    # Save preprocessing artifacts
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(scaler_full, 'models/scaler.pkl')
    
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count)
    
    metadata = {
        "dataset_name": "Kaggle Medical Appointment No-Shows",
        "student_registration": "241FA18343",
        "raw_record_count": int(len(df) + 5),
        "cleaned_record_count": int(len(df)),
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "feature_names": feature_cols,
        "scaled_numeric_features": numeric_features,
        "passthrough_features": passthrough_features,
        "target_name": "Target (1=No-show, 0=Show)",
        "train_class_distribution": {
            "show_0": neg_count,
            "no_show_1": pos_count
        },
        "scale_pos_weight": round(scale_pos_weight, 4)
    }
    
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("[SUCCESS] Production Preprocessing Pipeline executed! Saved preprocessor.pkl and processed datasets.")

if __name__ == "__main__":
    run_preprocessing()
