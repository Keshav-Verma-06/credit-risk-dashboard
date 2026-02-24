# 💳 Credit Risk Assessment Dashboard

A full-fledged enterprise-grade Streamlit dashboard for credit risk classification using machine learning.

**Author:** Keshav Verma  
**Student ID:** iitp_aiml_2506273  
**Project:** Credit Risk Classification System

---

## 🎯 Project Overview

This application helps financial institutions assess credit risk for loan applicants using advanced machine learning models. It provides:

- 🔮 **Single Prediction**: Assess individual applicants with instant risk scoring
- 📊 **Batch Prediction**: Upload CSV files to evaluate multiple applicants
- 📈 **Data Explorer**: Analyze patterns in credit data
- ⚙️ **Model Performance**: View detailed evaluation metrics

---

## 📁 Project Structure

```
Masai_proj/
│
├── app.py                      # Main Streamlit dashboard
├── logic.py                    # Core ML/Data processing functions
├── requirements.txt            # Python dependencies
├── .gitignore                 # Files to exclude from version control
├── README.md                  # This file
│
├── 01_eda_preprocessing.py    # Original Colab notebook (EDA)
├── 02_model_development.py    # Original Colab notebook (Model Training)
├── 03_model_evaluation.py     # Original Colab notebook (Evaluation)
│
├── data/                      # Dataset folder (add your CSV here)
├── Models/                    # Trained models (.pkl files)
├── Visualizations/            # Generated charts and plots
└── Processed_Data/           # Processed datasets
```

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd "c:\Users\Keshav Verma\Downloads\Masai_proj"

# Install required packages
pip install -r requirements.txt
```

### Step 2: Everything is Ready!

The dataset and trained model are already included in the repository:
- ✅ `data/german_credit_data.csv` - German Credit Dataset (1000 records)
- ✅ `Models/xgboost_model.pkl` - Pre-trained XGBoost model (81% accuracy)

**The dashboard auto-loads everything on startup - no uploads needed!**

### Step 3: Run Locally

```bash
# Run the Streamlit app
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 🌐 Deployment to Streamlit Cloud

### Option 1: Deploy via Streamlit Cloud (Recommended - FREE)

1. **Push to GitHub:**

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - Credit Risk Dashboard"

# Create a new repository on GitHub (https://github.com/new)
# Then connect and push:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-dashboard.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Log in with your GitHub account
   - Click "New App"
   - Select your repository: `credit-risk-dashboard`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Your app will be live at:**
   ```
   https://YOUR_USERNAME-credit-risk-dashboard.streamlit.app
   ```

### Important: Handling Data & Models

**❌ DO NOT commit large files to GitHub:**
- Large datasets (> 100 MB)
- Trained models (.pkl files > 100 MB)

**✅ Instead, use one of these methods:**

1. **Load data from URL** (Recommended for deployment):
   ```python
   # In the sidebar, use "Load from URL" option
   # Paste your Google Drive link
   ```

2. **Use GitHub Releases** for large models:
   - Upload models to GitHub Releases
   - Download in app using code

3. **Use cloud storage**:
   - AWS S3
   - Google Cloud Storage
   - Dropbox

---

## 🔒 Handling Secrets (API Keys, Credentials)

If you have API keys or database credentials:

1. **Create `.streamlit/secrets.toml` locally:**

```toml
# .streamlit/secrets.toml
[database]
username = "your_username"
password = "your_password"

[api]
key = "your_api_key"
```

2. **Access in code:**

```python
import streamlit as st

db_user = st.secrets["database"]["username"]
api_key = st.secrets["api"]["key"]
```

3. **Add secrets in Streamlit Cloud:**
   - Go to your app settings
   - Navigate to "Secrets"
   - Paste your secrets.toml content

---

## 📊 Using the Dashboard

### Dashboard Features

The dashboard automatically loads the trained model and dataset on startup. All features are ready to use immediately:

### 1. Single Prediction

- Navigate to "🔮 Single Prediction"
- Enter applicant details:
  - Age, Gender, Job Level
  - Housing status
  - Savings & Checking account status
  - Credit amount, Duration, Purpose
- Click "Assess Credit Risk"
- View detailed risk report

### 2. Batch Prediction

- Navigate to "📊 Batch Prediction"
- Upload CSV file with multiple applicants
- Required columns:
  ```
  Age, Sex, Job, Housing, Saving accounts, 
  Checking account, Credit amount, Duration, Purpose
  ```
- Click "Process All Predictions"
- Download results as CSV

### 3. Data Explorer

- Navigate to "📈 Data Explorer"
- Explore:
  - Statistical summaries
  - Risk distribution
  - Feature distributions
  - Correlation analysis

### 4. Model Performance

- Navigate to "⚙️ Model Performance"
- Click "Run Model Evaluation"
- View:
  - Confusion Matrix
  - Precision, Recall, ROC-AUC
  - Business interpretation

---

## 🛠️ Development & Customization

### Adding New Features

Edit `logic.py` to add custom feature engineering:

```python
def engineer_features(df):
    # Add your custom features here
    df['Custom_Feature'] = df['Feature1'] / df['Feature2']
    return df
```

### Customizing the UI

Edit `app.py` to modify the dashboard:

```python
# Change color scheme
st.set_page_config(
    page_icon="🏦",  # Change icon
    layout="wide"    # Change layout
)
```

### Training New Models

Use the original notebooks:
1. Run `01_eda_preprocessing.py` for data preparation
2. Run `02_model_development.py` for model training
3. Run `03_model_evaluation.py` for evaluation
4. Save best model to `Models/` folder
5. Upload to dashboard

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "Model file too large for GitHub"

**Solution:**
1. Add to `.gitignore`:
   ```
   Models/*.pkl
   ```
2. Use Git LFS or cloud storage

### Issue: "Streamlit Cloud deployment fails"

**Solution:**
1. Check `requirements.txt` has all dependencies
2. Ensure no absolute file paths in code
3. Use relative paths: `./data/file.csv`
4. Check app logs in Streamlit Cloud

### Issue: "Cannot access Google Drive link"

**Solution:**
1. Make drive link public ("Anyone with the link")
2. Convert to direct download link:
   ```
   https://drive.google.com/uc?export=download&id=FILE_ID
   ```

---

## 📚 Technical Details

### Machine Learning Pipeline

1. **Preprocessing:**
   - Missing value imputation
   - Feature engineering
   - StandardScaler for numeric features
   - OneHotEncoder for categorical features

2. **Models Supported:**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost (Primary)

3. **Evaluation Metrics:**
   - Recall (Bad class) - Primary metric
   - Precision, F1-Score
   - ROC-AUC
   - Confusion Matrix

### Key Business Logic

**High Risk Indicators:**
- Credit amount > 10,000
- Duration > 36 months
- Limited/unknown savings
- Age < 25 (limited history)
- High-risk loan purpose (discretionary spending)

**Low Risk Indicators:**
- Strong savings & checking accounts
- Home ownership
- Age 30-50 (established)
- Asset-backed loan purposes

---

## 📄 License

This project is created for educational purposes as part of IITP AIML coursework.

---

## 👨‍💼 Contact

**Developer:** Keshav Verma  
**Student ID:** iitp_aiml_2506273  
**Email:** [Your email if you want to share]  
**GitHub:** [Your GitHub username]

---

## ✅ Deployment Checklist

Before deploying to production:

- [ ] Test locally with `streamlit run app.py`
- [ ] Ensure all dependencies in `requirements.txt`
- [ ] Add `.gitignore` to exclude large files
- [ ] Remove hardcoded credentials
- [ ] Test with sample data
- [ ] Create GitHub repository
- [ ] Push to GitHub
- [ ] Deploy on Streamlit Cloud
- [ ] Test live URL
- [ ] Share with stakeholders

---

## 🎓 Credits

Based on Credit Risk Classification project for IITP AIML Program Term 2.

**Dataset:** German Credit Data  
**Models:** Scikit-learn, XGBoost  
**Dashboard:** Streamlit, Plotly

---

**🚀 Ready to Deploy? Follow the Quick Start Guide above!**

For issues or questions, refer to the Troubleshooting section or contact the developer.
