# Churn Prediction App

Streamlit app for unsupervised churn analysis using RFM (Recency, Frequency, Monetary), hybrid clustering (KMeans + Agglomerative), and SHAP explainability.

How to run

1. Create and activate virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

2. Install requirements:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

Usage

- Upload a CSV in the sidebar and map columns if names differ.
- Select number of clusters in the sidebar.
- Explore tabs: RFM, distributions, clustering, churn prediction (cluster-based), SHAP explanations.

Notes

- For large files, consider sampling before upload.
- SHAP can be slow; reduce sample size in the sidebar.

License: MIT (change as needed)
