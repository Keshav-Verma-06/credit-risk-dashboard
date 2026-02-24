"""
Credit Risk Classification - Core Logic Module
This module contains all data processing, feature engineering, and model prediction functions.
Extracted and refactored from Colab notebooks for production deployment.

Author: Keshav Verma
Student ID: iitp_aiml_2506273
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Feature definitions
NUMERIC_FEATURES = ['Age', 'Credit amount', 'Duration', 'Credit_to_Duration']
CATEGORICAL_FEATURES = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age_Group']


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(file_path=None, url=None):
    """
    Load credit risk dataset from file or URL.
    
    Args:
        file_path: Local path to CSV file
        url: URL to download CSV from (Google Drive link will be converted)
        
    Returns:
        pandas DataFrame
    """
    try:
        if url:
            # Convert Google Drive sharing link to direct download link
            if 'drive.google.com' in url:
                file_id = url.split('/d/')[1].split('/')[0] if '/d/' in url else None
                if file_id:
                    url = f'https://drive.google.com/uc?export=download&id={file_id}'
            df = pd.read_csv(url)
        elif file_path:
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Either file_path or url must be provided")
        
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def preprocess_data(df):
    """
    Clean and preprocess the credit risk dataset.
    
    Steps:
    1. Handle missing values
    2. Feature engineering
    3. Encode target variable
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Processed DataFrame
    """
    df_processed = df.copy()
    
    # Handle missing values - impute with "Unknown" for financial account features
    for col in ['Saving accounts', 'Checking account']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
    
    # Feature Engineering
    df_processed = engineer_features(df_processed)
    
    # Encode target variable if it's categorical
    if 'Risk' in df_processed.columns:
        if df_processed['Risk'].dtype == 'object':
            df_processed['Risk'] = df_processed['Risk'].map({'Good': 0, 'Bad': 1})
    
    return df_processed


def engineer_features(df):
    """
    Create business-value features from raw data.
    
    New Features:
    - Credit_to_Duration: Monthly payment proxy
    - Age_Group: Life-stage segmentation
    - Liquidity_Flag: Financial resilience indicator
    - High_Risk_Purpose: Discretionary spending flag
    
    Args:
        df: DataFrame with raw features
        
    Returns:
        DataFrame with engineered features
    """
    df_eng = df.copy()
    
    # 1. Credit_to_Duration ratio (monthly payment proxy)
    if 'Credit amount' in df_eng.columns and 'Duration' in df_eng.columns:
        df_eng['Credit_to_Duration'] = df_eng['Credit amount'] / (df_eng['Duration'] + 1)
    
    # 2. Age_Group bins
    if 'Age' in df_eng.columns:
        age_bins = [17, 25, 35, 50, 100]
        age_labels = ['18-25', '26-35', '36-50', '50+']
        df_eng['Age_Group'] = pd.cut(df_eng['Age'], bins=age_bins, labels=age_labels, right=True)
        df_eng['Age_Group'] = df_eng['Age_Group'].astype(str)
    
    # 3. Liquidity_Flag
    if 'Saving accounts' in df_eng.columns and 'Checking account' in df_eng.columns:
        high_saving = df_eng['Saving accounts'].isin(['quite rich', 'rich'])
        high_checking = df_eng['Checking account'].isin(['rich'])
        df_eng['Liquidity_Flag'] = (high_saving & high_checking).astype(int)
    
    # 4. High_Risk_Purpose
    if 'Purpose' in df_eng.columns:
        high_risk_purposes = ['radio/TV', 'repairs', 'furniture', 'domestic appliances']
        df_eng['High_Risk_Purpose'] = df_eng['Purpose'].isin(high_risk_purposes).astype(int)
    
    return df_eng


def create_preprocessing_pipeline():
    """
    Create sklearn preprocessing pipeline for model deployment.
    
    Returns:
        ColumnTransformer object
    """
    # Numeric pipeline: Standardization
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: One-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            drop=None
        ))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])
    
    return preprocessor


# ============================================================================
# MODEL PREDICTION
# ============================================================================

def load_model(model_path):
    """
    Load trained model from pickle file.
    
    Args:
        model_path: Path to .pkl file
        
    Returns:
        Trained model object
    """
    try:
        import joblib
        import sklearn
        from packaging import version
        
        # Load model with joblib (better for sklearn models)
        model_data = joblib.load(model_path)
        
        # Handle both old pickle format and new metadata format
        if isinstance(model_data, dict) and 'model' in model_data:
            # New format with metadata
            trained_version = model_data.get('sklearn_version', 'unknown')
            current_version = sklearn.__version__
            
            # Warn if version mismatch
            if trained_version != 'unknown' and trained_version != current_version:
                import streamlit as st
                st.warning(f"⚠️ sklearn version mismatch: Model trained with {trained_version}, running {current_version}")
            
            return model_data['model']
        else:
            # Old format - just the model
            return model_data
            
    except AttributeError as e:
        if "_RemainderColsList" in str(e) or "sklearn" in str(e).lower():
            raise Exception(f"sklearn version incompatibility detected. Please retrain the model with current sklearn version. Original error: {str(e)}")
        raise Exception(f"Error loading model: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def predict_credit_risk(model, input_data):
    """
    Make credit risk prediction on new applicant data.
    
    Args:
        model: Trained model pipeline
        input_data: DataFrame or dict with applicant features
        
    Returns:
        Dictionary with prediction and probability
    """
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Preprocess input
    input_processed = preprocess_data(input_data)
    
    # Make prediction
    prediction = model.predict(input_processed)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_processed)[0]
        risk_proba = proba[1]  # Probability of Bad class
    else:
        risk_proba = None
    
    result = {
        'prediction': 'Bad (High Risk)' if prediction == 1 else 'Good (Low Risk)',
        'prediction_label': int(prediction),
        'risk_probability': risk_proba
    }
    
    return result


def batch_predict(model, df):
    """
    Make predictions on multiple applicants at once.
    
    Args:
        model: Trained model pipeline
        df: DataFrame with applicant data
        
    Returns:
        DataFrame with predictions added
    """
    df_processed = preprocess_data(df)
    predictions = model.predict(df_processed)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(df_processed)[:, 1]
    else:
        probabilities = None
    
    df_result = df.copy()
    df_result['Predicted_Risk'] = ['Bad' if p == 1 else 'Good' for p in predictions]
    
    if probabilities is not None:
        df_result['Risk_Probability'] = probabilities
    
    return df_result


# ============================================================================
# ANALYSIS & STATISTICS
# ============================================================================

def get_data_statistics(df):
    """
    Calculate key statistics and insights from the dataset.
    
    Args:
        df: DataFrame
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_records': len(df),
        'numeric_summary': df.select_dtypes(include=[np.number]).describe().to_dict(),
        'categorical_summary': {}
    }
    
    # Risk distribution if available
    if 'Risk' in df.columns:
        risk_counts = df['Risk'].value_counts()
        stats['risk_distribution'] = risk_counts.to_dict()
        stats['default_rate'] = (df['Risk'] == 1).mean() * 100 if df['Risk'].dtype in [int, float] else None
    
    # Categorical summaries
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        stats['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    return stats


def calculate_risk_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate business-critical metrics for credit risk models.
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_good': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'precision_bad': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_good': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'recall_bad': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # ROC-AUC if probabilities available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Business interpretation
    if cm.shape == (2, 2):
        total_defaults = cm[1, 0] + cm[1, 1]
        caught_defaults = cm[1, 1]
        metrics['defaults_caught'] = caught_defaults
        metrics['total_defaults'] = total_defaults
        metrics['defaults_missed'] = cm[1, 0]
    
    return metrics


def generate_risk_report(applicant_data, prediction_result):
    """
    Generate a detailed risk assessment report for an applicant.
    
    Args:
        applicant_data: Dictionary with applicant features
        prediction_result: Dictionary from predict_credit_risk()
        
    Returns:
        Formatted string report
    """
    risk_level = prediction_result['prediction']
    risk_prob = prediction_result.get('risk_probability', 'N/A')
    
    if risk_prob != 'N/A':
        risk_prob_pct = f"{risk_prob * 100:.1f}%"
    else:
        risk_prob_pct = 'N/A'
    
    report = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 CREDIT RISK ASSESSMENT REPORT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🏷️  RISK CLASSIFICATION: {risk_level}
    📈 DEFAULT PROBABILITY: {risk_prob_pct}
    
    📋 APPLICANT PROFILE:
    {'─' * 44}
    """
    
    # Add key features
    key_features = ['Age', 'Credit amount', 'Duration', 'Purpose', 
                   'Saving accounts', 'Checking account', 'Housing']
    
    for feature in key_features:
        if feature in applicant_data:
            report += f"   • {feature:20s}: {applicant_data[feature]}\n"
    
    # Risk interpretation
    report += f"\n{'─' * 44}\n"
    report += "💡 BUSINESS INTERPRETATION:\n"
    
    if 'Bad' in risk_level:
        report += """   ⚠️  HIGH RISK - Consider:
      - Enhanced due diligence review
      - Higher interest rate to compensate
      - Reduced loan amount approval
      - Collateral requirements
    """
    else:
        report += """   ✅ LOW RISK - Recommended:
      - Standard approval process
      - Competitive interest rates
      - Full loan amount consideration
      - Premium customer treatment
    """
    
    report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    return report


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_input_data(input_data, required_features):
    """
    Validate that input data contains all required features.
    
    Args:
        input_data: DataFrame or dict
        required_features: List of required feature names
        
    Returns:
        Boolean and error message
    """
    if isinstance(input_data, dict):
        missing = [f for f in required_features if f not in input_data]
    else:
        missing = [f for f in required_features if f not in input_data.columns]
    
    if missing:
        return False, f"Missing required features: {', '.join(missing)}"
    
    return True, "Validation passed"


def save_processed_data(df, output_path):
    """
    Save processed data to CSV or pickle.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    if output_path.endswith('.pkl'):
        df.to_pickle(output_path)
    else:
        df.to_csv(output_path, index=False)
    
    return output_path


# ============================================================================
# CONSTANTS FOR STREAMLIT DROPDOWNS
# ============================================================================

FEATURE_OPTIONS = {
    'Sex': ['male', 'female'],
    'Housing': ['own', 'rent', 'free'],
    'Saving accounts': ['little', 'moderate', 'quite rich', 'rich', 'Unknown'],
    'Checking account': ['little', 'moderate', 'rich', 'Unknown'],
    'Purpose': ['car', 'furniture', 'radio/TV', 'education', 'business', 
                'domestic appliances', 'repairs', 'vacation/others'],
    'Job': [0, 1, 2, 3]
}


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'load_data',
    'preprocess_data',
    'engineer_features',
    'create_preprocessing_pipeline',
    'load_model',
    'predict_credit_risk',
    'batch_predict',
    'get_data_statistics',
    'calculate_risk_metrics',
    'generate_risk_report',
    'validate_input_data',
    'save_processed_data',
    'FEATURE_OPTIONS',
    'NUMERIC_FEATURES',
    'CATEGORICAL_FEATURES'
]
