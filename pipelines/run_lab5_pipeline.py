"""
Lab 5 Pipeline Runner: Production Data Pipeline & Schema Validation
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 5: Production Data Pipeline")
    print("[INFO] Project: Medical Appointment No-Show Prediction")
    print("[INFO] Registration: 241FA18343")
    print("[INFO] =========================================")
    
    scripts = [
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py"
    ]
    
    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} caught an error.")
            sys.exit(1)
            
    print("\n[SUCCESS] Lab 5 Production Data Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
