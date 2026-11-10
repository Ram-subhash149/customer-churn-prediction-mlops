"""
Data Validation Module
Project: Medical Appointment No-Show Prediction
Student Registration: 241FA18343
"""

import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

def get_no_show_schema():
    """Defines the strict schema specification for the Kaggle Medical Appointment No-Show dataset."""
    return pa.DataFrameSchema({
        "PatientId": Column(pa.Float, Check.gt(0)),
        "AppointmentID": Column(pa.Int, Check.gt(0)),
        "Gender": Column(pa.String, Check.isin(["F", "M"])),
        "ScheduledDay": Column(pa.String),
        "AppointmentDay": Column(pa.String),
        "Age": Column(pa.Int, Check.ge(-1)),  # Raw data has 1 known anomaly of -1, filtered during preprocessing
        "Neighbourhood": Column(pa.String),
        "Scholarship": Column(pa.Int, Check.isin([0, 1])),
        "Hipertension": Column(pa.Int, Check.isin([0, 1])),
        "Diabetes": Column(pa.Int, Check.isin([0, 1])),
        "Alcoholism": Column(pa.Int, Check.isin([0, 1])),
        "Handcap": Column(pa.Int, Check.isin([0, 1, 2, 3, 4])),
        "SMS_received": Column(pa.Int, Check.isin([0, 1])),
        "No-show": Column(pa.String, Check.isin(["Yes", "No"]))
    }, strict=True)

def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema and checks for data quality."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_no_show_schema()
    
    validation_passed = True
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema structure and type validation PASSED.")
    except pa.errors.SchemaErrors as err:
        print("[WARNING] Schema validation flagged anomalies:")
        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.to_string())
        
        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        validation_passed = False

    # Check for negative age anomaly
    neg_age_count = (df['Age'] < 0).sum()
    if neg_age_count > 0:
        print(f"[INFO] Data Quality Check: Detected {neg_age_count} record(s) with Age < 0 (will be cleaned in preprocessing).")

    # Check missing values
    missing_count = df.isna().sum().sum()
    if missing_count > 0:
        print(f"[WARNING] Missing values detected: {missing_count}")
    else:
        print("[SUCCESS] Zero missing values detected.")

    # Check duplicate rows
    dup_count = df.duplicated().sum()
    print(f"[INFO] Fully duplicated rows: {dup_count}")

    print("[SUCCESS] Data Validation completed successfully!")
    return validation_passed

if __name__ == "__main__":
    data_path = 'data/raw/KaggleV2-May-2016.csv'
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        validate_schema(raw_df, output_report_name="baseline_validation.csv")
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
        exit(1)
