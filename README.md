# Telecom churn predictor — deployment guide

This project has three parts, matching your flowchart:

```
model/     -> train.py, retrains the model exactly like your notebook
              (with the monthly_charges fix), saves churn_pipeline.joblib
backend/   -> FastAPI app that serves predictions (already includes the
              trained .joblib file, ready to deploy)
frontend/  -> index.html, a single-page site ("Signal") that calls the API
```

The model is already trained and bundled into `backend/churn_pipeline.joblib`.
You don't need to retrain anything unless you get new data — `model/train.py`
is there for that day.

## 1. Deploy the backend on Render

1. Create a free account at [render.com](https://render.com).
2. Push the `backend/` folder to a GitHub repo (or the whole project — Render
   just needs a path to the Dockerfile).
3. In Render: **New +** → **Web Service** → connect your repo.
4. Render will detect the `Dockerfile` automatically. Settings:
   - **Root directory:** `backend` (if you pushed the whole project)
   - **Instance type:** Free
5. Click **Deploy**. Render will build the image and give you a URL like:
   `https://churn-api-xxxx.onrender.com`
6. Test it once it's live:
   ```
   curl https://churn-api-xxxx.onrender.com/health
   ```
   You should see `{"status":"healthy","model_version":"1.0.0"}`.

Note: Render's free tier spins the service down after inactivity, so the
first request after a while takes ~30-50 seconds to wake up. That's normal.

## 2. Point the frontend at your backend

Open `frontend/index.html` and change this one line near the bottom:

```js
const API_BASE_URL = "http://localhost:8000";
```

to your real Render URL:

```js
const API_BASE_URL = "https://churn-api-xxxx.onrender.com";
```

## 3. Deploy the frontend

Easiest options, either works with zero config since it's a single static file:

**Vercel**
1. Push `frontend/` to GitHub (or drag-and-drop the folder at vercel.com/new).
2. Deploy — no build settings needed.

**Netlify**
1. Go to [app.netlify.com/drop](https://app.netlify.com/drop).
2. Drag the `frontend` folder in. Done — you get a live URL immediately.

## 4. (Optional) Tighten CORS

Right now the backend accepts requests from any origin (`allow_origins=["*"]`)
so you can test freely. Once your frontend has a fixed URL, you can lock it
down in `backend/main.py`:

```python
allow_origins=["https://your-frontend-url.vercel.app"],
```

Redeploy the backend after this change.

## Testing locally first (recommended)

Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
Then open `frontend/index.html` directly in a browser (API_BASE_URL should
stay as `http://localhost:8000`) and try a prediction.

## What the model expects

| Field | Type | Notes |
|---|---|---|
| gender | "Male" / "Female" | |
| age | number (18-100) | |
| tenure_months | number | months with the company |
| contract_type | "Month-To-Month" / "One Year" / "Two Year" | |
| internet_service | "Fiber" / "Dsl" / "No" | |
| num_addon_services | number | |
| monthly_charges | number ($) | |
| data_usage_gb | number | |
| support_calls | number | last cycle |
| payment_method | "Mailed Check" / "Bank Transfer" / "Credit Card" / "Electronic Check" | |
| total_charges | number ($) | lifetime total |

The API automatically derives two engineered features (`avg_monthly_spend`,
`is_new_customer`) the same way the notebook did — you don't need to send
them.
