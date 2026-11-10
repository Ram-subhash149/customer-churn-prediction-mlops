"""
Baseline Training Module
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def run_training():
    print("[INFO] Starting Baseline Model Training for Medical Appointment No-Show...")
    
    # 1. Load processed training data
    X_train_final = np.load('data/processed/X_train_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    # 2. Initialize and train the baseline Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    )
    
    print(f"[INFO] Fitting RandomForest on {X_train_final.shape[0]} patient appointment records...")
    model.fit(X_train_final, y_train)
    
    # 3. Save the trained baseline model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_baseline.pkl')
    print("[SUCCESS] Baseline model training complete! Saved to models/random_forest_baseline.pkl.")

if __name__ == "__main__":
    run_training()
