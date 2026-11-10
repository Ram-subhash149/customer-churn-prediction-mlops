# Medical Appointment No-Show Prediction

**Student Registration Number:** `241FA18343`  
**Target Variable:** `Target` (`1` = Patient did not show up, `0` = Patient showed up)  
**Dataset:** `KaggleV2-May-2016.csv` (~110,000 records from Brazilian public health clinics)  

---

## 1. Project Overview & Problem Statement

Missing medical appointments ("no-shows") results in substantial operational inefficiencies, inflated healthcare costs, and delayed medical care for patients on waiting lists. This project develops an end-to-end Machine Learning and MLOps system to predict the likelihood that a patient will fail to attend a scheduled medical appointment.

By providing calibrated probability scores prior to appointment dates, clinics can implement targeted retention strategies (automated SMS reminders, confirmation phone calls, overbooking protocols) to maximize resource utilization and improve patient health outcomes.

---

## 2. Project Architecture & Directory Structure

```text
No_Show_Appointments-241FA18343/
│
├── configs/
│   └── config.yaml                       # Centralized pipeline configurations
│
├── data/
│   ├── raw/
│   │   └── KaggleV2-May-2016.csv         # Raw appointment records (110,527 rows)
│   └── processed/
│       ├── X_train_final.npy              # Final feature matrix (Train: 88,416 rows)
│       ├── X_test_final.npy               # Final feature matrix (Test: 22,105 rows)
│       ├── X_train_scaled.npy             # Standardized feature matrix (Train)
│       ├── X_test_scaled.npy              # Standardized feature matrix (Test)
│       ├── y_train.npy                    # Binary target vector (Train)
│       ├── y_test.npy                     # Binary target vector (Test)
│       └── dataset_metadata.json          # Preprocessing lineage & metadata
│
├── notebooks/
│   └── no_show_appointments_project.ipynb # End-to-end research & EDA notebook
│
├── pipelines/
│   ├── run_lab3_baseline.py               # Lab 3: Preprocessing, Baseline Training, Evaluation
│   ├── run_lab4_tracking.py               # Lab 4: MLflow Tracking, Tuning & Reproducibility
│   ├── run_lab5_pipeline.py               # Lab 5: Schema Validation & Modular Pipeline
│   └── run_lab6_registry.py               # Lab 6: Model Registry & Lifecycle Management
│
├── src/
│   ├── validate_data.py                   # Pandera data contract & schema validation
│   ├── preprocess.py                      # Standalone data cleaning & feature engineering
│   ├── preprocess_pipeline.py             # Modular Scikit-Learn ColumnTransformer pipeline
│   ├── validate_outputs.py                # Preprocessing output & tensor integrity validator
│   ├── train.py                           # Baseline Random Forest training
│   ├── train_mlflow.py                    # Multi-model benchmarking & RandomizedSearchCV tuning
│   ├── train_registry.py                  # Model registration with MLflow Model Registry
│   ├── evaluate.py                        # Model evaluation & error analysis (FP / FN export)
│   ├── validate_reproducibility.py        # Deterministic seed reproducibility validator
│   ├── automate_lifecycle.py              # Champion-Challenger automated lifecycle promotion
│   └── generate_registry_report.py        # Production deployment JSON report generator
│
├── models/
│   ├── scaler.pkl                         # StandardScaler object
│   ├── preprocessor.pkl                   # Scikit-Learn ColumnTransformer pipeline
│   ├── random_forest_baseline.pkl         # Baseline Random Forest model
│   └── best_xgboost_model.pkl             # Tuned XGBoost production model
│
├── artifacts/
│   ├── roc_curve.png                      # Multi-model ROC comparison curve
│   ├── confusion_matrix.png               # Tuned XGBoost confusion matrix
│   ├── feature_importance.png             # XGBoost feature importance rankings
│   ├── threshold_analysis.png             # Precision / Recall / F1 vs. Threshold curves
│   ├── precision_recall_curve.png         # Precision-Recall curve with optimal F1 marker
│   ├── classification_report.json         # Detailed precision/recall/F1 metrics
│   ├── preprocessing_summary_report.json  # Data dimensions and quality checks
│   ├── production_model_report.json       # Production model lineage and stage metadata
│   └── model_metrics.json                 # Comprehensive empirical metrics
│
├── outputs/
│   ├── predictions.csv                    # Test predictions with probability scores
│   ├── false_negatives.csv                # Missed no-shows (Actual=1, Predicted=0)
│   └── false_positives.csv                # False alarms (Actual=0, Predicted=1)
│
├── reports/
│   └── model_comparison.csv              # Empirical model performance comparison table
│
├── logs/                                  # Execution logs
├── mlruns/                                # Local MLflow file artifacts
├── mlflow.db                              # SQLite MLflow tracking backend database
├── requirements.txt                       # Project dependencies
└── README.md                              # Comprehensive project documentation
```

---

## 3. Dataset & Data Cleaning

The dataset consists of **110,527** raw records from public clinics in Vitória, Brazil:
- **Missing values:** 0 missing values across all columns.
- **Duplicates:** 0 fully duplicate records.
- **Age correction:** 1 record with `Age == -1` was removed (`Age >= 0`).
- **Typographical corrections:** `Hipertension -> Hypertension`, `Handcap -> Handicap`.
- **Datetime conversions:** `ScheduledDay` and `AppointmentDay` converted to UTC datetime objects.
- **Negative waiting days:** 5 records where `ScheduledDay` occurred after `AppointmentDay` were removed (`WaitingDays >= 0`).
- **Cleaned dataset size:** **110,521** records.

---

## 4. Feature Engineering & Zero-Leakage Policy

All patient history features are calculated **strictly chronologically** after sorting by `['PatientId', 'AppointmentDay_dt', 'ScheduledDay_dt']`. Future appointments never influence earlier records:

1. **`WaitingDays`:** Lead time in calendar days between scheduling and appointment.
2. **`ScheduledWeekday`:** Day of the week when the appointment was scheduled (0 = Monday, 6 = Sunday).
3. **`AppointmentWeekday`:** Day of the week of the actual medical appointment.
4. **`ScheduledHour`:** Hour of the day when scheduling occurred.
5. **`PriorAppointments`:** Cumulative count of past appointments for the individual patient (`df.groupby('PatientId').cumcount()`).
6. **`PriorNoShows`:** Cumulative count of past no-shows for that patient prior to the current appointment (`df.groupby('PatientId')['Target'].cumsum() - df['Target']`).
7. **`PriorNoShowRate`:** Historical no-show proportion (`PriorNoShows / PriorAppointments`).
8. **`Gender`:** Binary mapped (`F -> 0`, `M -> 1`).

### Predictive Feature Set (15 Features)
```python
feature_cols = [
    'Gender', 'Age', 'Scholarship', 'Hypertension', 'Diabetes',
    'Alcoholism', 'Handicap', 'SMS_received', 'WaitingDays',
    'ScheduledWeekday', 'AppointmentWeekday', 'ScheduledHour',
    'PriorAppointments', 'PriorNoShows', 'PriorNoShowRate'
]
```

---

## 5. Machine Learning Models & Class Imbalance

The dataset is imbalanced: **79.81%** attended (`0`) and **20.19%** missed (`1`). Class imbalance is addressed across all models:
- **Logistic Regression:** `class_weight='balanced'` with `StandardScaler`.
- **Decision Tree:** `class_weight='balanced'`.
- **Random Forest:** `class_weight='balanced'`.
- **Gradient Boosting:** Sample weights assigned inversely proportional to class frequencies.
- **LightGBM:** `is_unbalance=True`.
- **XGBoost:** `scale_pos_weight = neg_count / pos_count = 3.95`.

### Hyperparameter Tuning
XGBoost was tuned using `RandomizedSearchCV` (40 iterations, 5-fold `StratifiedKFold`, optimizing for `ROC-AUC`).

---

## 6. Empirical Evaluation Results

All models were evaluated on the held-out stratified test set (22,105 appointments):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.6079** | **0.3160** | **0.8086** | **0.4544** | **0.7473** |
| **XGBoost (Default)** | 0.6071 | 0.3153 | 0.8075 | 0.4536 | 0.7466 |
| **LightGBM** | 0.6107 | 0.3162 | 0.7983 | 0.4530 | 0.7466 |
| **Gradient Boosting** | 0.6038 | 0.3140 | 0.8122 | 0.4529 | 0.7455 |
| **Random Forest** | 0.5889 | 0.3098 | 0.8443 | 0.4533 | 0.7438 |
| **Decision Tree** | 0.6101 | 0.3143 | 0.7878 | 0.4493 | 0.7319 |
| **Logistic Regression** | 0.6615 | 0.3176 | 0.5891 | 0.4127 | 0.6868 |

*Tuned XGBoost achieved a **+6.05 percentage point** gain in ROC-AUC over the baseline Logistic Regression.*

---

## 7. Threshold Optimization

Using the Precision-Recall curve of the tuned XGBoost model:
- **Default Threshold (0.50):** Precision = `0.3160`, Recall = `0.8086`, F1 = `0.4544`.
- **Optimal Threshold (`0.5503`):** Precision = `0.3393`, Recall = `0.7015`, F1 = **`0.4574`**, Accuracy = **`0.6640`**.

---

## 8. Top Feature Importance (XGBoost Tuned)

1. **`WaitingDays` (39.50%):** Lead time between booking and appointment is by far the strongest predictor.
2. **`PriorNoShows` (11.33%):** History of previous unexcused absences.
3. **`SMS_received` (10.30%):** Reminder message transmission status.
4. **`PriorNoShowRate` (6.76%):** Proportion of historical appointments missed.
5. **`Age` (4.75%):** Patient demographic vulnerability.

---

## 9. MLOps Lifecycle & Model Registry

The project implements a complete automated MLOps lifecycle:
1. **Experiment Tracking:** Runs, parameters, and metrics logged to SQLite-backed MLflow (`sqlite:///mlflow.db`).
2. **Model Registry:** The production candidate is versioned under `No_Show_Production_Model`.
3. **Automated Lifecycle Manager (`automate_lifecycle.py`):** Automatically evaluates staging models against the current Production champion on `ROC-AUC`, promoting the superior model and archiving defeated versions.
4. **Lineage:** The `models/preprocessor.pkl` pipeline is logged alongside model weights for reproducibility.

---

## 10. How to Run the Project

### Installation
```bash
pip install -r requirements.txt
```

### Option A: Run Individual Pipelines
```bash
# Lab 5: Production Data Pipeline & Schema Validation
python pipelines/run_lab5_pipeline.py

# Lab 3: Baseline Preprocessing, Training & Evaluation
python pipelines/run_lab3_baseline.py

# Lab 4: MLflow Tracking, All Models, XGBoost Tuning & Reproducibility
python pipelines/run_lab4_tracking.py

# Lab 6: Model Registry & Automated Lifecycle Management
python pipelines/run_lab6_registry.py
```

### Option B: Run Standalone Python Modules
```bash
# 1. Validate raw data contract
python src/validate_data.py

# 2. Execute preprocessing pipeline
python src/preprocess_pipeline.py

# 3. Validate preprocessed output tensors
python src/validate_outputs.py

# 4. Train baseline Random Forest
python src/train.py

# 5. Run full model comparison & MLflow tracking
python src/train_mlflow.py

# 6. Evaluate and export error analysis (FP / FN)
python src/evaluate.py

# 7. Validate deterministic reproducibility
python src/validate_reproducibility.py

# 8. Register production model
python src/train_registry.py

# 9. Automate champion-challenger promotion
python src/automate_lifecycle.py

# 10. Generate deployment report
python src/generate_registry_report.py
```

### Option C: Jupyter Notebook
Open and run `notebooks/no_show_appointments_project.ipynb` in any Jupyter interface or VS Code.


---

## 11. Run the Prediction API (Local)

The project now includes a FastAPI serving layer using the saved tuned XGBoost
model (`models/best_xgboost_model.pkl`).

From the project root in VS Code PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-api.txt
python -m uvicorn api.main:app --reload
```

Open:
- API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Model metadata: `http://127.0.0.1:8000/metadata`

The API derives the same 15 model features documented in this README from
appointment demographics, scheduled/appointment timestamps, and prior
appointment history. It uses the F1-optimised threshold of `0.5503` by default.

### Example prediction input

Use this JSON in Swagger UI under `POST /predict`:

```json
{
  "Gender": "F",
  "Age": 35,
  "Scholarship": 0,
  "Hypertension": 0,
  "Diabetes": 0,
  "Alcoholism": 0,
  "Handicap": 0,
  "SMS_received": 1,
  "ScheduledDay": "2016-05-01T10:30:00",
  "AppointmentDay": "2016-05-10T00:00:00",
  "PriorAppointments": 2,
  "PriorNoShows": 1,
  "threshold": 0.5503
}
```

`prediction = 1` means predicted no-show; `prediction = 0` means likely to
attend. This is a model estimate for operational decision support, not a
clinical decision.

## 12. Docker Deployment (Windows)

Ensure Docker Desktop is open and its Engine is running. From the project root:

```powershell
docker build -t no-show-prediction-api:1.0 .
docker run -d --name no-show-api -p 8000:8000 no-show-prediction-api:1.0
docker ps
```

Open `http://localhost:8000/docs`.

To inspect logs:

```powershell
docker logs no-show-api
```

To stop and remove the container:

```powershell
docker stop no-show-api
docker rm no-show-api
```

Alternative using Docker Compose:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f
```

Stop Compose:

```powershell
docker compose down
```

### API and Docker files

- `api/main.py` — FastAPI endpoints and feature construction.
- `requirements-api.txt` — serving dependencies.
- `Dockerfile` — container image definition.
- `docker-compose.yml` — one-command container orchestration.
- `.dockerignore` — excludes datasets, notebooks, local MLflow runs and logs from the image.
- `api/README.md` — concise API reference.
