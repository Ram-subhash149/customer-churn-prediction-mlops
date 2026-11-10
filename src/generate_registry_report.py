"""
Model Registry Deployment Report Generator
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient

def generate_registry_report():
    print("[INFO] Generating Model Registry Production Report for Medical Appointments...")
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    model_name = "No_Show_Production_Model"
    
    try:
        # Search for model in Production stage
        production_models = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in production_models if mv.current_stage == "Production"), None)
        
        if not champion:
            print("[WARNING] No model currently in Production stage! Searching latest available version...")
            if production_models:
                champion = production_models[-1]
            else:
                print("[ERROR] No registered versions found.")
                return
                
        # Get run details
        run = client.get_run(champion.run_id)
        
        report = {
            "project": "Medical Appointment No-Show Prediction",
            "student_registration": "241FA18343",
            "registry_status": "READY_FOR_CLINICAL_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": {
                "roc_auc": round(float(run.data.metrics.get("roc_auc", 0.0)), 4),
                "f1_score": round(float(run.data.metrics.get("f1_score", 0.0)), 4),
                "recall": round(float(run.data.metrics.get("recall", 0.0)), 4),
                "precision": round(float(run.data.metrics.get("precision", 0.0)), 4),
                "accuracy": round(float(run.data.metrics.get("accuracy", 0.0)), 4)
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl"
        }
        
        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/production_model_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        print(f"[SUCCESS] Production report generated for Version {champion.version} (Stage: {champion.current_stage})")
        print(f"[INFO] Saved to: {report_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")

if __name__ == "__main__":
    generate_registry_report()
