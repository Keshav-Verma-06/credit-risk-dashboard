"""
Deployment Debug Script
Check environment compatibility and model loading
"""
import streamlit as st
import sys
import os

st.title("🔍 Deployment Debug Info")

# Environment Info
st.header("1️⃣ Environment")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Python")
    st.code(sys.version)
    st.write(f"**Version:** {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

with col2:
    st.subheader("Working Directory")
    st.code(os.getcwd())

# Package Versions
st.header("2️⃣ Package Versions")
packages = {}
try:
    import sklearn
    packages['scikit-learn'] = sklearn.__version__
except Exception as e:
    packages['scikit-learn'] = f"ERROR: {e}"

try:
    import pandas as pd
    packages['pandas'] = pd.__version__
except Exception as e:
    packages['pandas'] = f"ERROR: {e}"

try:
    import numpy as np
    packages['numpy'] = np.__version__
except Exception as e:
    packages['numpy'] = f"ERROR: {e}"

try:
    import xgboost
    packages['xgboost'] = xgboost.__version__
except Exception as e:
    packages['xgboost'] = f"ERROR: {e}"

try:
    import joblib
    packages['joblib'] = joblib.__version__
except Exception as e:
    packages['joblib'] = f"ERROR: {e}"

try:
    import streamlit as st_ver
    packages['streamlit'] = st_ver.__version__
except Exception as e:
    packages['streamlit'] = f"ERROR: {e}"

for pkg, ver in packages.items():
    st.write(f"**{pkg}:** `{ver}`")

# File System Check
st.header("3️⃣ File System")
files_to_check = [
    'data/german_credit_data.csv',
    'Models/xgboost_model.pkl',
    'app.py',
    'logic.py',
    'rebuild_model.py',
    'requirements.txt'
]

for file_path in files_to_check:
    exists = os.path.exists(file_path)
    if exists:
        size = os.path.getsize(file_path) / 1024  # KB
        st.success(f"✅ {file_path} ({size:.2f} KB)")
    else:
        st.error(f"❌ {file_path} - NOT FOUND")

# Try Loading Model
st.header("4️⃣ Model Loading Test")
model_path = 'Models/xgboost_model.pkl'

if os.path.exists(model_path):
    try:
        import joblib
        st.info("Attempting to load model with joblib...")
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            st.success("✅ Model loaded successfully! (New format with metadata)")
            st.json({
                'sklearn_version': model_data.get('sklearn_version', 'N/A'),
                'trained_at': model_data.get('trained_at', 'N/A'),
                'python_version': model_data.get('python_version', 'N/A')[:50] + '...'
            })
        else:
            st.success("✅ Model loaded successfully! (Old format)")
            st.write(f"Model type: {type(model_data)}")
            
    except AttributeError as e:
        st.error(f"❌ AttributeError (sklearn version mismatch): {e}")
        st.warning("Solution: Delete model and rebuild, or retrain with current sklearn version")
    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__}: {e}")
        st.exception(e)
else:
    st.warning("⚠️ Model file not found. Will rebuild on app start.")

# Recommendations
st.header("5️⃣ Recommendations")
if packages.get('scikit-learn', '').startswith('ERROR'):
    st.error("🔴 sklearn not installed or import failed")
elif packages.get('scikit-learn') != '1.3.2':
    st.warning(f"⚠️ sklearn version {packages.get('scikit-learn')} differs from recommended 1.3.2")
    st.info("Consider pinning sklearn==1.3.2 in requirements.txt")
else:
    st.success("✅ sklearn version matches requirements")

if sys.version_info >= (3, 13):
    st.warning("⚠️ Python 3.13+ detected. Consider using Python 3.11 for better stability")
    st.info("Create .streamlit/python_version with content: 3.11")
elif sys.version_info < (3, 9):
    st.warning("⚠️ Python version too old. Upgrade to 3.11")
else:
    st.success("✅ Python version OK")
