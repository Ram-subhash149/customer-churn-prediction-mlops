"""
MLflow Experiment Tracking & Comprehensive Model Development
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb

def run_mlflow_pipeline():
    print("=" * 65)
    print("[INFO] Starting MLflow Medical Appointment No-Show Pipeline")
    print("=" * 65)
    
    # 1. Setup MLflow Tracking
    db_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(db_uri)
    experiment_name = "Medical_Appointment_No_Show_Prediction"
    mlflow.set_experiment(experiment_name)
    print(f"[INFO] MLflow tracking URI: {db_uri}")
    print(f"[INFO] MLflow experiment: {experiment_name}")
    
    # 2. Load Processed Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    X_train_scaled = np.load('data/processed/X_train_scaled.npy')
    X_test_scaled = np.load('data/processed/X_test_scaled.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    with open('data/processed/dataset_metadata.json', 'r') as f:
        metadata = json.load(f)
    feature_cols = metadata['feature_names']
    scale_pos_weight = metadata['scale_pos_weight']
    
    print(f"[INFO] Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"[INFO] Class balance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    results = {}
    
    # Helper to evaluate and track
    def evaluate_and_track(name, model, is_scaled=False, params=None):
        X_tr = X_train_scaled if is_scaled else X_train
        X_te = X_test_scaled if is_scaled else X_test
        
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        print(f"  -> {name:<26} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # MLflow Run logging
        run_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            mlflow.log_param("model_name", name)
            mlflow.log_param("is_scaled", is_scaled)
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc
            })
            try:
                mlflow.sklearn.log_model(sk_model=model, name="model", serialization_format="cloudpickle")
            except Exception as e:
                print(f"    [WARN] Model artifact logging skipped: {e}")
            if os.path.exists('data/processed/dataset_metadata.json'):
                mlflow.log_artifact('data/processed/dataset_metadata.json', artifact_path="metadata")
    
    # -------------------------------------------------------------
    # 3. Model 1: Logistic Regression (Baseline Linear Model)
    # -------------------------------------------------------------
    print("\n[INFO] Training Model 1: Logistic Regression...")
    lr_params = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}
    lr = LogisticRegression(**lr_params)
    lr.fit(X_train_scaled, y_train)
    evaluate_and_track("Logistic Regression", lr, is_scaled=True, params=lr_params)
    
    # -------------------------------------------------------------
    # 4. Model 2: Decision Tree
    # -------------------------------------------------------------
    print("\n[INFO] Training Model 2: Decision Tree...")
    dt_params = {"max_depth": 8, "class_weight": "balanced", "random_state": 42}
    dt = DecisionTreeClassifier(**dt_params)
    dt.fit(X_train, y_train)
    evaluate_and_track("Decision Tree", dt, is_scaled=False, params=dt_params)
    
    # -------------------------------------------------------------
    # 5. Model 3: Random Forest
    # -------------------------------------------------------------
    print("\n[INFO] Training Model 3: Random Forest...")
    rf_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    evaluate_and_track("Random Forest", rf, is_scaled=False, params=rf_params)
    
    # -------------------------------------------------------------
    # 6. Model 4: Gradient Boosting (sklearn)
    # -------------------------------------------------------------
    print("\n[INFO] Training Model 4: Gradient Boosting...")
    sample_weight_train = np.where(y_train == 1, scale_pos_weight, 1.0)
    gb_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "random_state": 42
    }
    gb = GradientBoostingClassifier(**gb_params)
    gb.fit(X_train, y_train, sample_weight=sample_weight_train)
    evaluate_and_track("Gradient Boosting", gb, is_scaled=False, params=gb_params)
    
    # -------------------------------------------------------------
    # 7. Model 5: XGBoost (Default Boosting)
    # -------------------------------------------------------------
    print("\n[INFO] Training Model 5: XGBoost...")
    xgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)
    evaluate_and_track("XGBoost", xgb_model, is_scaled=False, params=xgb_params)
    
    # -------------------------------------------------------------
    # 8. Model 6: LightGBM
    # -------------------------------------------------------------
    print("\n[INFO] Training Model 6: LightGBM...")
    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)
    evaluate_and_track("LightGBM", lgb_model, is_scaled=False, params=lgb_params)
    
    # -------------------------------------------------------------
    # 9. Model 7: XGBoost Hyperparameter Tuning with RandomizedSearchCV
    # -------------------------------------------------------------
    print("\n[INFO] Running XGBoost Hyperparameter Tuning via RandomizedSearchCV (n_iter=40, 5-fold CV)...")
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.19),
        'max_depth': randint(3, 9),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0.5, 4.5),
        'min_child_weight': randint(1, 20)
    }
    base_xgb = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_dist,
        n_iter=40,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    print(f"[SUCCESS] Best CV ROC-AUC: {search.best_score_:.4f}")
    print("[INFO] Best Hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
        
    evaluate_and_track("XGBoost (Tuned)", best_xgb, is_scaled=False, params=search.best_params_)
    
    # Save best XGBoost model
    joblib.dump(best_xgb, 'models/best_xgboost_model.pkl')
    
    # -------------------------------------------------------------
    # 10. Model Comparison Table & Report
    # -------------------------------------------------------------
    comparison_data = []
    for name, res in results.items():
        comparison_data.append({
            "Model": name,
            "Accuracy": round(res['accuracy'], 4),
            "Precision": round(res['precision'], 4),
            "Recall": round(res['recall'], 4),
            "F1_Score": round(res['f1'], 4),
            "ROC_AUC": round(res['roc_auc'], 4)
        })
    comp_df = pd.DataFrame(comparison_data).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    
    os.makedirs('reports', exist_ok=True)
    comp_df.to_csv('reports/model_comparison.csv', index=False)
    
    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 65)
    print(comp_df.to_string(index=False))
    print("=" * 65)
    
    # -------------------------------------------------------------
    # 11. Plot ROC Curves for all Models
    # -------------------------------------------------------------
    os.makedirs('artifacts', exist_ok=True)
    fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(results))
    
    for (name, res), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax_roc.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {res['roc_auc']:.4f})")
        
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
    ax_roc.set_title('ROC Curves - All Medical Appointment Models', fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', frameon=True)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    
    roc_img_path = 'artifacts/roc_curve.png'
    fig_roc.savefig(roc_img_path, dpi=300, bbox_inches='tight')
    plt.close(fig_roc)
    print(f"[INFO] Saved combined ROC curve to {roc_img_path}")
    
    # -------------------------------------------------------------
    # 12. Confusion Matrix for Best Model (XGBoost Tuned)
    # -------------------------------------------------------------
    best_res = results['XGBoost (Tuned)']
    cm = confusion_matrix(y_test, best_res['y_pred'])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Show (0)', 'No-Show (1)'],
                yticklabels=['Show (0)', 'No-Show (1)'],
                cbar=False)
    ax_cm.set_title('Confusion Matrix - XGBoost (Tuned)', fontsize=13, fontweight='bold')
    ax_cm.set_xlabel('Predicted Label', fontsize=11)
    ax_cm.set_ylabel('Actual Label', fontsize=11)
    cm_img_path = 'artifacts/confusion_matrix.png'
    fig_cm.savefig(cm_img_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    print(f"[INFO] Saved confusion matrix to {cm_img_path}")
    
    # -------------------------------------------------------------
    # 13. Feature Importance for Best Model
    # -------------------------------------------------------------
    importances = best_xgb.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    
    fig_fi, ax_fi = plt.subplots(figsize=(9, 6))
    feat_imp.plot(kind='barh', ax=ax_fi, color='#1f77b4', edgecolor='black')
    ax_fi.set_title('Feature Importance - XGBoost (Tuned)', fontsize=13, fontweight='bold')
    ax_fi.set_xlabel('Relative Importance (Gain)', fontsize=11)
    fi_img_path = 'artifacts/feature_importance.png'
    fig_fi.savefig(fi_img_path, dpi=300, bbox_inches='tight')
    plt.close(fig_fi)
    print(f"[INFO] Saved feature importance plot to {fi_img_path}")
    
    # -------------------------------------------------------------
    # 14. Threshold Optimization
    # -------------------------------------------------------------
    y_prob_best = best_res['y_prob']
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_prob_best)
    
    # F1 score across thresholds
    f1_arr = 2 * (prec_arr[:-1] * rec_arr[:-1]) / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)
    best_thresh_idx = int(np.argmax(f1_arr))
    best_thresh = float(thresh_arr[best_thresh_idx])
    best_f1_opt = float(f1_arr[best_thresh_idx])
    
    print(f"\n[INFO] Threshold Optimization:")
    print(f"  -> Optimal Threshold maximizing F1: {best_thresh:.4f}")
    print(f"  -> Peak F1-Score at optimal threshold: {best_f1_opt:.4f}")
    
    fig_th, axes_th = plt.subplots(1, 2, figsize=(13, 4.5))
    axes_th[0].plot(thresh_arr, prec_arr[:-1], label='Precision', color='#1f77b4', lw=2)
    axes_th[0].plot(thresh_arr, rec_arr[:-1], label='Recall', color='#ff7f0e', lw=2)
    axes_th[0].plot(thresh_arr, f1_arr, label='F1 Score', color='green', lw=2)
    axes_th[0].axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold = {best_thresh:.3f}')
    axes_th[0].set_xlabel('Classification Threshold', fontsize=11)
    axes_th[0].set_ylabel('Score', fontsize=11)
    axes_th[0].set_title('Precision, Recall & F1 vs. Threshold', fontsize=12, fontweight='bold')
    axes_th[0].legend(loc='best')
    axes_th[0].grid(True, linestyle='--', alpha=0.5)
    
    axes_th[1].plot(rec_arr, prec_arr, color='#1f77b4', lw=2, label='PR Curve')
    axes_th[1].scatter(rec_arr[best_thresh_idx], prec_arr[best_thresh_idx], color='red', s=90, zorder=5, label=f'Optimal (F1={best_f1_opt:.3f})')
    axes_th[1].set_xlabel('Recall', fontsize=11)
    axes_th[1].set_ylabel('Precision', fontsize=11)
    axes_th[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes_th[1].legend(loc='lower left')
    axes_th[1].grid(True, linestyle='--', alpha=0.5)
    
    fig_th.tight_layout()
    th_img_path = 'artifacts/threshold_analysis.png'
    fig_th.savefig(th_img_path, dpi=300, bbox_inches='tight')
    plt.close(fig_th)
    
    # Also save separate precision_recall_curve.png
    fig_pr, ax_pr = plt.subplots(figsize=(6.5, 5))
    ax_pr.plot(rec_arr, prec_arr, color='#1f77b4', lw=2, label='PR Curve')
    ax_pr.scatter(rec_arr[best_thresh_idx], prec_arr[best_thresh_idx], color='red', s=90, zorder=5, label=f'Optimal (Threshold={best_thresh:.3f})')
    ax_pr.set_xlabel('Recall', fontsize=11)
    ax_pr.set_ylabel('Precision', fontsize=11)
    ax_pr.set_title('Precision-Recall Curve - Best XGBoost Model', fontsize=12, fontweight='bold')
    ax_pr.legend(loc='lower left')
    ax_pr.grid(True, linestyle='--', alpha=0.5)
    pr_img_path = 'artifacts/precision_recall_curve.png'
    fig_pr.savefig(pr_img_path, dpi=300, bbox_inches='tight')
    plt.close(fig_pr)
    print(f"[INFO] Saved threshold analysis plots to {th_img_path} and {pr_img_path}")
    
    # Metrics with optimal threshold
    y_pred_opt = (y_prob_best >= best_thresh).astype(int)
    clf_rep_dict = classification_report(y_test, y_pred_opt, target_names=['Show (0)', 'No-Show (1)'], output_dict=True)
    with open('artifacts/classification_report.json', 'w') as f:
        json.dump(clf_rep_dict, f, indent=4)
        
    model_metrics = {
        "project": "Medical Appointment No-Show Prediction",
        "student_registration": "241FA18343",
        "best_model_name": "XGBoost (Tuned)",
        "default_threshold_metrics": {
            "accuracy": round(best_res['accuracy'], 4),
            "precision": round(best_res['precision'], 4),
            "recall": round(best_res['recall'], 4),
            "f1_score": round(best_res['f1'], 4),
            "roc_auc": round(best_res['roc_auc'], 4)
        },
        "optimized_threshold_metrics": {
            "optimal_threshold": round(best_thresh, 4),
            "accuracy": round(accuracy_score(y_test, y_pred_opt), 4),
            "precision": round(precision_score(y_test, y_pred_opt, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_opt, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred_opt, zero_division=0), 4),
            "roc_auc": round(best_res['roc_auc'], 4)
        },
        "top_features_ranked": feat_imp.sort_values(ascending=False).to_dict()
    }
    with open('artifacts/model_metrics.json', 'w') as f:
        json.dump(model_metrics, f, indent=4)
        
    print("[SUCCESS] All models trained, evaluated, tracked in MLflow, and artifacts generated!")

if __name__ == "__main__":
    run_mlflow_pipeline()
