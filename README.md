# SalarySense AI — upgraded project

Features: modern multi-tab UI; 25+ job roles in demo; upload CSV; Linear Regression and Random Forest comparison; MAE/RMSE/R²; optional age/city/industry/company-size/work-mode features when present in CSV; salary graph and five-year compounded scenario update after prediction.

IMPORTANT: the default 1,800 rows are synthetic demonstration data, not real Kaggle observations. Upload a real Kaggle CSV for final evaluation. Required columns: experience, education, job role, salary. Extra profile fields affect predictions only when they exist in the uploaded dataset.

## Run on Windows
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Public deployment (works without your Wi-Fi or PC)
1. Create a GitHub repository at https://github.com/new
2. Upload app.py, requirements.txt and README.md to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Click Create app; choose the repository, branch main, and file path app.py.
5. Click Deploy. Share the generated https://....streamlit.app URL.
6. Test with phone Wi-Fi turned off/mobile data. The cloud URL runs on Streamlit servers; the local 192.168.x.x:8501 address only works on your local network while the PC and app are running.
7. To update the site, commit the revised files to GitHub; Streamlit redeploys from the repository.

Verify the Kaggle dataset's headers, licence, currency, salary period, and date range before using it in the final report.
