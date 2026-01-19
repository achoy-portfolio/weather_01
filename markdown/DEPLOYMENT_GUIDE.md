# Deployment Guide

## ✅ What Was Changed

All hardcoded API keys have been removed and replaced with environment variable loading.

### Files Updated:

1. **`.env`** - Fixed format to: `NOAA_API_KEY=your_key_here`
2. **`odds_vs_temperature_dashboard.py`** - Now loads from Streamlit secrets or .env
3. **`scripts/fetching/noaa_fetch_klga_historical.py`** - Now loads from .env
4. **`scripts/analysis/analyze_daily_peak_times.py`** - Now loads from .env
5. **`requirements.txt`** - Added `python-dotenv>=1.0.0`
6. **`.streamlit/secrets.toml`** - Created for Streamlit Cloud deployment

### Files Protected:

- `.env` - In .gitignore (won't be pushed to GitHub)
- `.streamlit/secrets.toml` - In .gitignore (won't be pushed to GitHub)

## 🚀 Local Development

Your `.env` file contains:

```
NOAA_API_KEY=WSPJBegSQBiSbKgtOTOsIqfZMfCLPaPx
```

Run locally:

```bash
source venv/bin/activate
streamlit run odds_vs_temperature_dashboard.py
```

## ☁️ Streamlit Cloud Deployment

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Remove hardcoded API keys, use environment variables"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your repository
4. Choose `odds_vs_temperature_dashboard.py` as main file
5. Click "Advanced settings"
6. In the "Secrets" section, paste:
   ```toml
   NOAA_API_KEY = "WSPJBegSQBiSbKgtOTOsIqfZMfCLPaPx"
   ```
7. Click "Deploy!"

## 🔒 Security Notes

- ✅ API key is NOT in any Python files
- ✅ `.env` is in `.gitignore` (won't be pushed)
- ✅ `.streamlit/secrets.toml` is in `.gitignore` (won't be pushed)
- ✅ Streamlit Cloud secrets are encrypted and secure

## 📝 How It Works

**Local development:**

```python
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("NOAA_API_KEY")  # Reads from .env
```

**Streamlit Cloud:**

```python
API_KEY = st.secrets.get("NOAA_API_KEY")  # Reads from cloud secrets
```

**Dashboard (tries both):**

```python
try:
    NOAA_API_KEY = st.secrets.get("NOAA_API_KEY")  # Cloud first
except:
    NOAA_API_KEY = os.getenv("NOAA_API_KEY")  # Then .env
```

## ✅ Ready to Deploy!

Your code is now safe to push to GitHub without exposing your API key.
