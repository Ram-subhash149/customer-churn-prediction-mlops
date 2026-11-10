"""
Lab 6 Pipeline Runner: Model Registry & Automated Lifecycle Management
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 6: Model Registry & Lifecycle Management")
    print("[INFO] Project: Medical Appointment No-Show Prediction")
    print("[INFO] Registration: 241FA18343")
    print("[INFO] =========================================")
    
    scripts = [
        "src/train_registry.py",          # Trains and registers candidate in MLflow Registry
        "src/automate_lifecycle.py",      # Compares candidate against Production Champion and transitions stages
        "src/generate_registry_report.py" # Generates JSON production deployment report
    ]
    
    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} caught an error.")
            sys.exit(1)
            
    print("\n[SUCCESS] Lab 6 Model Registry & Lifecycle Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
