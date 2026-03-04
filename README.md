# Heart Failure Risk Demo — Deployment Guide

This repository contains a small Streamlit app that loads a trained logistic regression model and a scaler to provide a probabilistic estimate of heart disease risk from user-entered features.

This README explains how to run the app locally on Windows (PowerShell), prepare a minimal requirements file, and basic options for cloud or container deployment.

## Files you should have in the project root

- `app.py` — Streamlit app (entrypoint)
- `heart_model_lr.joblib` — trained logistic regression model
- `scaler.joblib` — StandardScaler fitted on training data (exposes `feature_names_in_`)
- `heart_model_calibrator.joblib` — (optional) plain calibrator dict created by calibration script
- `heart.csv` — dataset used for calibration/testing (not required to run the app but useful for retraining)

If any of the model artifacts are missing, the app may fall back to the uncalibrated model or fail with a helpful error.

## Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended)
- PowerShell (Windows) — commands below use PowerShell syntax

## Recommended Python packages

Install the core packages needed to run the app. You can create a virtual environment first (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install streamlit scikit-learn pandas numpy joblib
```

Optional (for plotting or extended diagnostics):

```powershell
pip install matplotlib seaborn
```

You can create a minimal `requirements.txt` for deployment like:

```
streamlit
scikit-learn
pandas
numpy
joblib
matplotlib
seaborn
```

Generate an exact `requirements.txt` from your environment:

```powershell
pip freeze > requirements.txt
```

## Run the app locally (PowerShell)

Activate your virtual environment and run Streamlit from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run .\app.py --server.port 8503
```

Notes:
- If port 8503 is already in use, choose another port (e.g., `--server.port 8504`).
- If the app edits files, re-run `streamlit run` or use Streamlit's auto-reload.

To run Streamlit in the background on Windows PowerShell (detach):

```powershell
Start-Process -NoNewWindow -FilePath "streamlit" -ArgumentList "run .\app.py --server.port 8503"
```

## Quick sanity checks

- If the app fails to unpickle model files, ensure `scikit-learn` and `joblib` are installed and that the sklearn version is compatible with the version used to create the artifacts. You may see an `InconsistentVersionWarning` when versions differ — usually the model still loads but test predictions to confirm behavior.
- Confirm these files are present in the same folder as `app.py`: `scaler.joblib`, `heart_model_lr.joblib`, and (optionally) `heart_model_calibrator.joblib`.

## Deploying to Streamlit Cloud (recommended for quick demos)

1. Push this repository to GitHub.
2. Create an account on https://streamlit.io/cloud and connect your GitHub.
3. Choose the repo and the branch, set the command to `streamlit run app.py`, and optionally set the Python version and `requirements.txt`.

Streamlit Cloud will build and serve your app automatically.

## Docker (optional)

Here's a minimal `Dockerfile` you can use as a starting point. Create the file in the project root and build an image.

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

Build and run (PowerShell):

```powershell
docker build -t heart-app .
docker run -p 8501:8501 heart-app
```

## Troubleshooting

- If `app.py` raises an IndentationError or SyntaxError after an edit, run:

```powershell
python -m py_compile .\app.py
```

- If model unpickling fails with version warnings or errors, try installing the scikit-learn version the model was trained with (if known) e.g.: `pip install scikit-learn==1.5.2` and re-run.
- If Streamlit shows nothing in the browser, ensure the correct port is open in your firewall and that you used the correct server port when starting Streamlit.

## Security and privacy note

This project uses a public dataset and ships model artifacts for local demo purposes. Do not use it for real medical decisions. Do not upload or process private patient data without appropriate safeguards and approvals.

## Contact / Next steps

- If you'd like, I can:
  - Add a `requirements.txt` to the repo.
  - Add a `Dockerfile` and `.dockerignore` so you can build an image quickly.
  - Add a tiny GitHub Actions workflow to auto-deploy to Streamlit Cloud on push.

Enjoy — open `http://localhost:8503` (or your chosen port) after starting the app.

## Storing large model artifacts: Git LFS

If your model artifact files (for example `*.joblib`) are large, it's best to store them with Git LFS to avoid bloating your Git history.

Quick steps (PowerShell):

```powershell
# Install Git LFS (one-time)
choco install git-lfs -y   # if you use Chocolatey on Windows
# or follow instructions from https://git-lfs.github.com/

git lfs install
git lfs track "*.joblib"
git add .gitattributes
git add path\to\heart_model_lr.joblib path\to\scaler.joblib
git commit -m "Add model artifacts tracked via Git LFS"
git push origin main
```

Notes:
- After running `git lfs track` the `.gitattributes` file is updated — commit and push it so others pulling the repo will also get LFS settings.
- If you don't have Chocolatey, install Git LFS from https://git-lfs.github.com/ and follow the platform-specific instructions.

