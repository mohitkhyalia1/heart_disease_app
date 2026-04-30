# Heart Disease App

A heart disease risk prediction web app built with Streamlit, using the UCI Cleveland Heart Disease dataset. The app includes interactive risk prediction, dataset overview charts, and feature importance visualizations.

## What this repo contains

- `app.py` — Streamlit web application for interactive prediction and analysis
- `train_and_save.py` — training script that downloads the dataset, trains a Random Forest model, and saves model artifacts
- `heart.csv` — dataset file (local copy)
- `requirements.txt` — Python package dependencies

## Run locally

1. Create a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model and save artifacts:

```powershell
python train_and_save.py
```

This creates the `models/best_model.pkl` and `models/scaler.pkl` files required by `app.py`.

4. Launch the website locally:

```powershell
streamlit run app.py
```

Then open the URL printed by Streamlit (usually `http://localhost:8501`).

## Deploy as a website

### Option 1: Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app in Streamlit Cloud.
3. Set the app path to `app.py`.
4. Add `requirements.txt` and the `models/` folder to the repo.

### Option 2: Any Python hosting platform

Use a platform that supports Python web apps, install dependencies, and run:

```powershell
streamlit run app.py
```

## Notes

- `app.py` already implements a complete web interface.
- If model files are missing, the app will show a warning and ask you to run `python train_and_save.py` first.
- This project is for educational use only and is not medical advice.

## Next steps

If you want, I can also help you:
- convert this into a Flask/HTML website
- add a simple landing page with navigation
- deploy the app to Streamlit Cloud or Render
