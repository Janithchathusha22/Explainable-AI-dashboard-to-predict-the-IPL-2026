# Deployment Guide

## Can This Be Hosted From Git?

Yes. Put the project in a GitHub repository, then connect that repository to Streamlit Community Cloud.

Use GitHub for source control and portfolio visibility. Use Streamlit Community Cloud for the live web app.

GitHub Pages is not the right target for this project because GitHub Pages serves static HTML/CSS/JavaScript, while this dashboard is a Python Streamlit app.

## Files Needed For Deployment

Keep these files in the GitHub repo:

- `streamlit_app.py`
- `requirements.txt`
- `requirements-ml.txt`
- `runtime.txt`
- `reports/`
- Input CSV files used by the dashboard

The dashboard reads already-generated report artifacts, so the live app can load quickly without retraining on every page visit.

`requirements.txt` is intentionally lightweight for cloud deployment. Use `requirements-ml.txt` locally when retraining the models.

## GitHub Setup

From the project folder:

```bash
git init
git add .
git commit -m "Initial IPL prediction dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

If Git says files are too large, use Git LFS for model artifacts or remove `models/` from the deployed app. The current Streamlit frontend only needs `reports/` and the input tables it displays.

## Streamlit Community Cloud Setup

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Select the repository.
4. Select branch: `main`.
5. Set main file path: `streamlit_app.py`.
6. Deploy.

After deployment, Streamlit gives a public URL like:

```text
https://your-app-name.streamlit.app
```

Use that URL in LinkedIn, your CV, and the GitHub README.

## Local Verification Before Deploy

Run:

```bash
python -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Retraining Workflow

When new data changes:

```bash
pip install -r requirements-ml.txt
python feature_engineering.py
python explainable_ai_pipeline.py
git add match_features.csv player_impact_scores.csv reports models
git commit -m "Update prediction artifacts"
git push
```

Streamlit Community Cloud will redeploy from the updated GitHub commit.

## LinkedIn Checklist

Before posting:

- Add live Streamlit URL to `README.md`.
- Add GitHub repo URL to the LinkedIn post.
- Add one screenshot of the dashboard.
- Mention the result, confidence level, and explainability method.
- Be clear that this is a prediction/explainability project, not betting advice.
