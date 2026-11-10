"""
Preprocessing Module
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    print("[INFO] Starting Medical Appointment No-Show Preprocessing...")
    
    # 1. Load Raw Dataset
    data_path = 'data/raw/KaggleV2-May-2016.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded raw dataset with shape: {df.shape}")
    
    # 2. Data Cleaning
    # Rename columns with spelling corrections
    df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
    
    # Remove records with invalid age (< 0)
    initial_rows = len(df)
    df = df[df['Age'] >= 0].copy()
    print(f"[INFO] Filtered negative age records. Removed: {initial_rows - len(df)} row(s).")
    
    # Convert dates to UTC datetime
    df['ScheduledDay_dt'] = pd.to_datetime(df['ScheduledDay'], utc=True)
    df['AppointmentDay_dt'] = pd.to_datetime(df['AppointmentDay'], utc=True)
    
    # 3. Feature Engineering
    # Calculate WaitingDays (normalized by day)
    df['WaitingDays'] = (
        df['AppointmentDay_dt'].dt.normalize() - df['ScheduledDay_dt'].dt.normalize()
    ).dt.days
    
    # Remove records with invalid negative waiting days
    before_wait_filter = len(df)
    df = df[df['WaitingDays'] >= 0].copy()
    print(f"[INFO] Filtered negative waiting day records. Removed: {before_wait_filter - len(df)} row(s).")
    
    # Extract date features
    df['ScheduledWeekday'] = df['ScheduledDay_dt'].dt.dayofweek
    df['AppointmentWeekday'] = df['AppointmentDay_dt'].dt.dayofweek
    df['ScheduledHour'] = df['ScheduledDay_dt'].dt.hour
    
    # Clean binary target: 1 = Did not show up (No-show: Yes), 0 = Showed up (No-show: No)
    df['Target'] = (df['No-show'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    
    # Sort chronologically BEFORE computing patient history to guarantee zero data leakage
    df = df.sort_values(['PatientId', 'AppointmentDay_dt', 'ScheduledDay_dt']).reset_index(drop=True)
    
    # Prior patient appointment history (strictly past appointments only)
    df['PriorAppointments'] = df.groupby('PatientId').cumcount()
    df['PriorNoShows'] = (df.groupby('PatientId')['Target'].cumsum() - df['Target']).astype(int)
    df['PriorNoShowRate'] = (df['PriorNoShows'] / df['PriorAppointments'].replace(0, 1)).astype(float)
    
    # Encode Gender: F -> 0, M -> 1
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1}).fillna(0).astype(int)
    
    # 4. Feature Selection
    feature_cols = [
        'Gender', 'Age', 'Scholarship', 'Hypertension', 'Diabetes',
        'Alcoholism', 'Handicap', 'SMS_received', 'WaitingDays',
        'ScheduledWeekday', 'AppointmentWeekday', 'ScheduledHour',
        'PriorAppointments', 'PriorNoShows', 'PriorNoShowRate'
    ]
    
    X = df[feature_cols].copy()
    y = df['Target'].copy()
    
    print(f"[INFO] Cleaned records: {len(df)}")
    print(f"[INFO] Target distribution: Show={int((y==0).sum())} ({((y==0).mean()*100):.2f}%), No-Show={int((y==1).sum())} ({((y==1).mean()*100):.2f}%)")
    
    # 5. Train-Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train_final = X_train.to_numpy(dtype=np.float64)
    X_test_final = X_test.to_numpy(dtype=np.float64)
    
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/X_train_scaled.npy', X_train_scaled)
    np.save('data/processed/X_test_scaled.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))
    
    joblib.dump(scaler, 'models/scaler.pkl')
    
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count)
    
    # Save Preprocessing Metadata
    metadata = {
        "dataset_name": "Kaggle Medical Appointment No-Shows",
        "student_registration": "241FA18343",
        "raw_record_count": int(initial_rows),
        "cleaned_record_count": int(len(df)),
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "feature_names": feature_cols,
        "target_name": "Target (1=No-show, 0=Show)",
        "train_class_distribution": {
            "show_0": neg_count,
            "no_show_1": pos_count
        },
        "scale_pos_weight": round(scale_pos_weight, 4)
    }
    
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[SUCCESS] Preprocessing completed successfully! Shapes: Train {X_train_final.shape}, Test {X_test_final.shape}")

if __name__ == "__main__":
    run_preprocessing()
