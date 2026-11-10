# Prediction API

Run locally from the project root:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Docker

```powershell
docker build -t no-show-prediction-api:1.0 .
docker run -d --name no-show-api -p 8000:8000 no-show-prediction-api:1.0
```

Or use Docker Compose:

```powershell
docker compose up --build -d
```

Stop:

```powershell
docker stop no-show-api
```

Remove:

```powershell
docker rm no-show-api
```

## API endpoints

- `GET /` — service information
- `GET /health` — model health
- `GET /metadata` — model and feature metadata
- `POST /predict` — one appointment
- `POST /predict/batch` — up to 500 appointments

The API accepts demographic indicators, scheduled and appointment timestamps,
and prior appointment history. It derives the 15 model features in the same
order documented in the research project. The default decision threshold is
0.5503, matching the project's reported F1-optimised threshold.

Example request:

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

`prediction: 1` means predicted no-show; `prediction: 0` means likely to attend.
The probability is a model estimate and should support, not replace, clinic
judgement. Do not use the demo API with identifiable real patient information.
