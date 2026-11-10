"""
Lab 3 Pipeline Runner: Baseline ML Pipeline
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 3: Baseline ML Pipeline")
    print("[INFO] Project: Medical Appointment No-Show Prediction")
    print("[INFO] Registration: 241FA18343")
    print("[INFO] =========================================")
    
    scripts = [
        "src/preprocess.py",
        "src/train.py",
        "src/evaluate.py"
    ]
    
    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed with exit code {result.returncode}.")
            sys.exit(1)
            
    print("\n[SUCCESS] Lab 3 Baseline Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
