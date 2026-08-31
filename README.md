# Pulse

Predict churn risk for telecom subscribers from plan, usage, and billing data.

Pulse is a small full-stack tool: a trained scikit-learn model served through a FastAPI backend, with a static frontend for entering a customer's profile and getting an instant churn prediction. Built for deployment on Vercel.

## How it works

1. Enter a customer's account details (contract type, tenure, internet service, usage, charges, support calls) in the form.
2. The frontend sends that data to the backend API.
3. A trained RandomForest model returns a churn probability, and the UI shows a risk tier (low / medium / high) with a short recommendation.

## Project structure

```
.
├── backend/
│   ├── api/
│   │   ├── main.py                # FastAPI app + prediction logic
│   │   └── churn_pipeline.joblib  # trained model (preprocessing + classifier)
│   └── requirements.txt
└── frontend/
    └── index.html                 # single-page form + result UI
```

## API

Deployed on Vercel, the backend exposes:

| Method | Path                  | Description                          |
|--------|------------------------|---------------------------------------|
| GET    | `/api/main`            | Health check / service info          |
| GET    | `/api/main/health`     | Model status                         |
| POST   | `/api/main/predict`    | Returns churn prediction for a customer |

**Request body for `/api/main/predict`:**

```json
{
  "gender": "Male",
  "age": 42,
  "tenure_months": 24,
  "contract_type": "Month-To-Month",
  "internet_service": "Fiber",
  "num_addon_services": 2,
  "monthly_charges": 65.00,
  "data_usage_gb": 95,
  "support_calls": 1,
  "payment_method": "Credit Card",
  "total_charges": 1560.00
}
```

**Response:**

```json
{
  "prediction": 1,
  "label": "Likely to churn",
  "confidence": 0.81,
  "churn_probability": 0.74
}
```

## Tech stack

- **Backend:** FastAPI, scikit-learn, pandas, joblib
- **Frontend:** Plain HTML/CSS/JS, no build step
- **Hosting:** Vercel (backend as a Python serverless function, frontend as a static site)

## Local development

**Backend:**

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload
```

The API will run at `http://localhost:8000`.

**Frontend:**

Open `frontend/index.html` in a browser, or serve it locally. Update `API_BASE_URL` in the `<script>` tag to point at your local or deployed backend.

## Deployment (Vercel)

This project deploys as **two separate Vercel projects** from the same repo:

1. Import the repo, set **Root Directory** to `backend`, deploy. Copy the resulting URL.
2. Update `API_BASE_URL` in `frontend/index.html` to that URL, commit.
3. Import the repo again, set **Root Directory** to `frontend`, deploy.

## License

MIT
