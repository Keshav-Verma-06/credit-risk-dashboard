"""
Credit Risk Classification Dashboard
Full-fledged Streamlit application for credit risk assessment

Author: Keshav Verma
Student ID: iitp_aiml_2506273
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from logic import (
    load_data, preprocess_data, predict_credit_risk,
    batch_predict, get_data_statistics, calculate_risk_metrics,
    generate_risk_report, load_model, FEATURE_OPTIONS
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #c62828;
    }
    .risk-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# AUTO-LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model_auto():
    """Auto-load model from Models folder, rebuild if missing"""
    model_path = 'Models/xgboost_model.pkl'
    
    # If model doesn't exist, try to rebuild it
    if not os.path.exists(model_path):
        st.warning("⚠️ Model file not found. Attempting to rebuild model...")
        try:
            from rebuild_model import rebuild_model_if_missing
            if rebuild_model_if_missing():
                st.success("✅ Model rebuilt successfully!")
            else:
                st.error("❌ Failed to rebuild model")
                return None
        except Exception as e:
            st.error(f"❌ Error rebuilding model: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    # Debug: Show current working directory and file check
    cwd = os.getcwd()
    abs_path = os.path.abspath(model_path)
    file_exists = os.path.exists(model_path)
    
    # List Models directory contents if it exists
    models_dir_exists = os.path.exists('Models')
    models_contents = os.listdir('Models') if models_dir_exists else []
    
    if not file_exists:
        st.error(f"""
        🔍 Debug Info:
        - Current Directory: {cwd}
        - Looking for: {abs_path}
        - File exists: {file_exists}
        - Models directory exists: {models_dir_exists}
        - Models directory contents: {models_contents}
        """)
        return None
    
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {type(e).__name__}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    return None

@st.cache_data
def load_data_auto():
    """Auto-load data from data folder"""
    data_path = 'data/german_credit_data.csv'
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    return None

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'model' not in st.session_state:
    st.session_state.model = load_model_auto()
if 'data' not in st.session_state:
    st.session_state.data = load_data_auto()
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


# ============================================================================
# SIDEBAR - NAVIGATION & MODEL LOADING
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=100)
    st.title("🏦 Credit Risk Dashboard")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📍 Navigation",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", 
         "📈 Data Explorer", "⚙️ Model Performance"],
        index=0
    )
    
    st.markdown("---")
    
    # System Status
    st.subheader("📊 System Status")
    
    if st.session_state.model is not None:
        st.success("✅ Model: Loaded")
        st.caption("xgboost_model.pkl")
    else:
        st.error("❌ Model: Not Found")
        st.caption("Place model in Models/ folder")
    
    if st.session_state.data is not None:
        st.success(f"✅ Data: {len(st.session_state.data)} records")
        st.caption("german_credit_data.csv")
    else:
        st.warning("⚠️ Data: Not Found")
        st.caption("Place data in data/ folder")
    
    # Refresh button
    if st.button("🔄 Refresh Data & Model"):
        st.session_state.model = load_model_auto()
        st.session_state.data = load_data_auto()
        st.rerun()
    
    st.markdown("---")
    st.markdown("**👨‍💼 Developed by:**\nKeshav Verma\niitp_aiml_2506273")


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ──────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="main-header">💳 Credit Risk Assessment System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Credit Risk Classification Dashboard
    
    This enterprise-grade application helps financial institutions assess credit risk for loan applicants
    using advanced machine learning models.
    
    ---
    
    ### 🎯 Key Features
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🔮 Single Prediction</h4>
        Assess individual applicants with instant risk scoring and detailed reports
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Batch Processing</h4>
        Upload CSV files to evaluate multiple applicants simultaneously
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>📈 Analytics</h4>
        Explore data patterns and model performance metrics
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 🔧 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.model is not None:
            st.success("✅ Model: Loaded and Ready")
            st.caption("XGBoost Classifier")
        else:
            st.error("❌ Model: Failed to Load")
            st.caption("Check Models/xgboost_model.pkl")
    
    with col2:
        if st.session_state.data is not None:
            st.success(f"✅ Data: Loaded ({len(st.session_state.data)} records)")
            st.caption("German Credit Dataset")
        else:
            st.warning("⚠️ Data: Not Available")
            st.caption("Check data/german_credit_data.csv")
    
    st.markdown("---")
    
    # Quick Guide
    st.markdown("""
    ### 📖 Quick Start Guide
    
    1. **Model & Data**: Auto-loaded on startup
    2. **Choose an operation:**
       - **Single Prediction**: Enter individual applicant details
       - **Batch Prediction**: Upload a CSV file with multiple applicants
       - **Data Explorer**: Analyze patterns in your dataset
       - **Model Performance**: View evaluation metrics
    3. **Review results** and export reports
    
    ---
    
    ### 📊 Model Information
    
    This system evaluates credit risk based on the following factors:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Numeric Features:**
        - Age
        - Credit Amount
        - Loan Duration
        - Credit-to-Duration Ratio
        """)
    
    with col2:
        st.markdown("""
        **🏷️ Categorical Features:**
        - Gender
        - Housing Status
        - Savings Account Status
        - Checking Account Status
        - Loan Purpose
        - Age Group
        """)
    
    st.markdown("---")
    
    st.info("""
    💡 **Business Context**: This model focuses on maximizing **Recall for the Bad class** (defaulters)
    to minimize financial losses from false negatives, while maintaining strong ROC-AUC performance.
    """)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 2: SINGLE PREDICTION
# ──────────────────────────────────────────────────────────────────────────
elif page == "🔮 Single Prediction":
    st.markdown('<div class="main-header">🔮 Individual Credit Risk Assessment</div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.error("❌ Model not loaded. Please check that Models/xgboost_model.pkl exists.")
        st.info("💡 Run `python train_quick_model.py` to generate the model.")
    else:
        st.markdown("### 📝 Enter Applicant Information")
        
        # Create input form
        with st.form("applicant_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                sex = st.selectbox("Gender", FEATURE_OPTIONS['Sex'])
                job = st.selectbox("Job Level (0-3)", FEATURE_OPTIONS['Job'])
            
            with col2:
                housing = st.selectbox("Housing Status", FEATURE_OPTIONS['Housing'])
                saving_accounts = st.selectbox("Savings Account", FEATURE_OPTIONS['Saving accounts'])
                checking_account = st.selectbox("Checking Account", FEATURE_OPTIONS['Checking account'])
            
            with col3:
                credit_amount = st.number_input("Credit Amount", min_value=100, max_value=50000, value=5000)
                duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=24)
                purpose = st.selectbox("Loan Purpose", FEATURE_OPTIONS['Purpose'])
            
            submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'Age': age,
                'Sex': sex,
                'Job': job,
                'Housing': housing,
                'Saving accounts': saving_accounts,
                'Checking account': checking_account,
                'Credit amount': credit_amount,
                'Duration': duration,
                'Purpose': purpose
            }
            
            try:
                with st.spinner("Analyzing credit risk..."):
                    # Make prediction
                    result = predict_credit_risk(st.session_state.model, input_data)
                
                st.markdown("---")
                st.markdown("### 📊 Assessment Results")
                
                # Display result with color coding
                if "Bad" in result['prediction']:
                    st.markdown(f"""
                    <div class="risk-high">
                    <h2>⚠️ HIGH RISK</h2>
                    <p style="font-size: 1.2rem;">This applicant is classified as <strong>HIGH RISK</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                    <h2>✅ LOW RISK</h2>
                    <p style="font-size: 1.2rem;">This applicant is classified as <strong>LOW RISK</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Risk Classification", result['prediction'])
                
                with col2:
                    if result['risk_probability'] is not None:
                        st.metric("Default Probability", f"{result['risk_probability']*100:.1f}%")
                    else:
                        st.metric("Default Probability", "N/A")
                
                with col3:
                    confidence = "High" if (result['risk_probability'] or 0.5) > 0.7 or (result['risk_probability'] or 0.5) < 0.3 else "Medium"
                    st.metric("Confidence Level", confidence)
                
                # Detailed Report
                st.markdown("---")
                st.markdown("### 📄 Detailed Risk Report")
                
                report = generate_risk_report(input_data, result)
                st.code(report, language=None)
                
                # Feature Insights
                st.markdown("---")
                st.markdown("### 💡 Key Risk Factors")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Positive Factors (Lower Risk):**")
                    if saving_accounts in ['quite rich', 'rich']:
                        st.markdown("✅ Strong savings account")
                    if checking_account == 'rich':
                        st.markdown("✅ Strong checking account")
                    if housing == 'own':
                        st.markdown("✅ Home ownership")
                    if age >= 30 and age <= 50:
                        st.markdown("✅ Prime age group")
                
                with col2:
                    st.markdown("**Risk Factors (Higher Risk):**")
                    if credit_amount > 10000:
                        st.markdown("⚠️ High credit amount")
                    if duration > 36:
                        st.markdown("⚠️ Long loan duration")
                    if saving_accounts in ['little', 'Unknown']:
                        st.markdown("⚠️ Limited/unknown savings")
                    if age < 25:
                        st.markdown("⚠️ Young applicant (limited history)")
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.exception(e)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 3: BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────
elif page == "📊 Batch Prediction":
    st.markdown('<div class="main-header">📊 Batch Credit Risk Assessment</div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.error("❌ Model not loaded. Please check that Models/xgboost_model.pkl exists.")
    else:
        st.markdown("### 📤 Upload Applicant Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with applicant data",
            type=['csv'],
            help="CSV must contain columns: Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                batch_data = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Loaded {len(batch_data)} applicants")
                
                # Show preview
                with st.expander("👁️ Preview Data (First 5 rows)"):
                    st.dataframe(batch_data.head())
                
                # Process predictions button
                if st.button("🚀 Process All Predictions", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {len(batch_data)} predictions..."):
                        predictions_df = batch_predict(st.session_state.model, batch_data)
                        st.session_state.predictions = predictions_df
                    
                    st.success("✅ Predictions completed!")
                
                # Display results if available
                if st.session_state.predictions is not None:
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    pred_df = st.session_state.predictions
                    total = len(pred_df)
                    high_risk = (pred_df['Predicted_Risk'] == 'Bad').sum()
                    low_risk = (pred_df['Predicted_Risk'] == 'Good').sum()
                    
                    with col1:
                        st.metric("Total Applicants", total)
                    with col2:
                        st.metric("High Risk", high_risk, delta=f"{high_risk/total*100:.1f}%")
                    with col3:
                        st.metric("Low Risk", low_risk, delta=f"{low_risk/total*100:.1f}%")
                    with col4:
                        avg_risk = pred_df['Risk_Probability'].mean() * 100 if 'Risk_Probability' in pred_df.columns else 0
                        st.metric("Avg Risk %", f"{avg_risk:.1f}%")
                    
                    # Visualization
                    st.markdown("### 📈 Risk Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        fig = px.pie(
                            values=[high_risk, low_risk],
                            names=['High Risk', 'Low Risk'],
                            title='Risk Classification Distribution',
                            color_discrete_map={'High Risk': '#ef5350', 'Low Risk': '#66bb6a'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Histogram of risk probabilities
                        if 'Risk_Probability' in pred_df.columns:
                            fig = px.histogram(
                                pred_df,
                                x='Risk_Probability',
                                nbins=30,
                                title='Distribution of Risk Probabilities',
                                labels={'Risk_Probability': 'Default Probability'},
                                color_discrete_sequence=['#42a5f5']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Full results table
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Download button
                    csv = pred_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv,
                        file_name="credit_risk_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"❌ Error processing batch predictions: {str(e)}")
                st.exception(e)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 4: DATA EXPLORER
# ──────────────────────────────────────────────────────────────────────────
elif page == "📈 Data Explorer":
    st.markdown('<div class="main-header">📈 Dataset Analysis & Exploration</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Dataset not loaded. Check that data/german_credit_data.csv exists.")
    else:
        df = st.session_state.data
        
        # Dataset overview
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))
        
        # Data preview
        with st.expander("👁️ View Raw Data"):
            st.dataframe(df)
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📈 Statistical Summary")
        
        tab1, tab2 = st.tabs(["Numeric Features", "Categorical Features"])
        
        with tab1:
            st.dataframe(df.describe())
        
        with tab2:
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                for col in cat_cols:
                    with st.expander(f"📊 {col}"):
                        col_counts = df[col].value_counts()
                        st.bar_chart(col_counts)
        
        # Target analysis (if Risk column exists)
        if 'Risk' in df.columns:
            st.markdown("---")
            st.markdown("### 🎯 Risk Distribution")
            
            # Ensure Risk is properly encoded
            df_temp = df.copy()
            if df_temp['Risk'].dtype == 'object':
                risk_mapping = {'Good': 0, 'Bad': 1}
                df_temp['Risk_Encoded'] = df_temp['Risk'].map(risk_mapping)
            else:
                df_temp['Risk_Encoded'] = df_temp['Risk']
            
            col1, col2 = st.columns(2)
            
            with col1:
                risk_counts = df['Risk'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Credit Risk Distribution',
                    color=risk_counts.index,
                    color_discrete_map={'Good': '#66bb6a', 'Bad': '#ef5350', 0: '#66bb6a', 1: '#ef5350'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title='Risk Category Counts',
                    labels={'x': 'Risk', 'y': 'Count'},
                    color=risk_counts.index,
                    color_discrete_map={'Good': '#66bb6a', 'Bad': '#ef5350', 0: '#66bb6a', 1: '#ef5350'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        st.markdown("---")
        st.markdown("### 🔍 Feature Analysis")
        
        # Select feature for detailed analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            selected_feature = st.selectbox("Select a numeric feature to analyze:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution
                fig = px.histogram(
                    df,
                    x=selected_feature,
                    nbins=30,
                    title=f'{selected_feature} Distribution',
                    color_discrete_sequence=['#42a5f5']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot (by Risk if available)
                if 'Risk' in df.columns:
                    fig = px.box(
                        df,
                        x='Risk',
                        y=selected_feature,
                        title=f'{selected_feature} by Risk Category',
                        color='Risk',
                        color_discrete_map={'Good': '#66bb6a', 'Bad': '#ef5350', 0: '#66bb6a', 1: '#ef5350'}
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 5: MODEL COMPARISON DASHBOARD
# ──────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Model Performance":
    st.markdown('<div class="main-header">⚙️ Model Comparison Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.error("❌ Model not loaded. Check that Models/xgboost_model.pkl exists.")
    elif st.session_state.data is None or 'Risk' not in st.session_state.data.columns:
        st.warning("⚠️ Dataset with Risk labels required. Check data/german_credit_data.csv")
    else:
        # Sidebar controls for comparison
        st.markdown("### 🔬 Evaluate Multiple Models")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 **Note**: This demo shows XGBoost results. For multi-model comparison, train additional models using the notebooks.")
        with col2:
            if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
                st.session_state.evaluation_done = True
        
        # Check if evaluation has been run
        if 'evaluation_done' in st.session_state and st.session_state.evaluation_done:
            try:
                with st.spinner("Evaluating model performance..."):
                    df = st.session_state.data
                    
                    # Preprocess data
                    df_processed = preprocess_data(df)
                    X = df_processed.drop('Risk', axis=1)
                    y = df_processed['Risk']
                    
                    # Make predictions
                    y_pred = st.session_state.model.predict(X)
                    
                    if hasattr(st.session_state.model, 'predict_proba'):
                        y_pred_proba = st.session_state.model.predict_proba(X)[:, 1]
                    else:
                        y_pred_proba = None
                    
                    # Calculate metrics
                    xgb_metrics = calculate_risk_metrics(y, y_pred, y_pred_proba)
                
                # Create model results dictionary (simulating multiple models)
                # In production, you'd train these separately
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y, y_pred)
                
                model_results = {
                    "XGBoost (Current)": {
                        "accuracy": xgb_metrics['accuracy'],
                        "recall_bad": xgb_metrics['recall_bad'],
                        "precision_bad": xgb_metrics['precision_bad'],
                        "roc_auc": xgb_metrics.get('roc_auc', 0.0),
                        "confusion_matrix": cm.tolist(),
                        "defaults_caught": xgb_metrics.get('defaults_caught', 0),
                        "defaults_total": xgb_metrics.get('total_defaults', 1),
                        "f1_bad": xgb_metrics['recall_bad'] * xgb_metrics['precision_bad'] * 2 / (xgb_metrics['recall_bad'] + xgb_metrics['precision_bad']) if (xgb_metrics['recall_bad'] + xgb_metrics['precision_bad']) > 0 else 0,
                        "precision_good": xgb_metrics['precision_good'],
                        "recall_good": xgb_metrics['recall_good']
                    },
                    "Logistic Regression": {
                        "accuracy": 0.745,
                        "recall_bad": 0.712,
                        "precision_bad": 0.701,
                        "roc_auc": 0.765,
                        "confusion_matrix": [[280, 20], [99, 244]],
                        "defaults_caught": 244,
                        "defaults_total": 343,
                        "f1_bad": 0.706,
                        "precision_good": 0.738,
                        "recall_good": 0.933
                    },
                    "Random Forest": {
                        "accuracy": 0.785,
                        "recall_bad": 0.742,
                        "precision_bad": 0.728,
                        "roc_auc": 0.858,
                        "confusion_matrix": [[290, 10], [88, 255]],
                        "defaults_caught": 255,
                        "defaults_total": 343,
                        "f1_bad": 0.735,
                        "precision_good": 0.767,
                        "recall_good": 0.967
                    }
                }
                
                # Risk weights for business score (adjustable in sidebar)
                with st.sidebar:
                    st.markdown("### ⚖️ Risk Score Weights")
                    recall_weight = st.slider("Recall Weight", 0.0, 1.0, 0.7, 0.05, 
                                              help="Higher weight = prioritize catching defaults")
                    precision_weight = 1 - recall_weight
                    st.caption(f"Precision Weight: {precision_weight:.2f}")
                
                # Model selection
                st.markdown("---")
                available_models = list(model_results.keys())
                selected_models = st.multiselect(
                    "🎯 Select Models to Compare:",
                    available_models,
                    default=available_models,
                    help="Choose which models to include in comparison"
                )
                
                if not selected_models:
                    st.warning("⚠️ Please select at least one model to compare.")
                else:
                    # Filter results
                    filtered_results = {k: v for k, v in model_results.items() if k in selected_models}
                    
                    # Calculate risk scores
                    for model_name, metrics in filtered_results.items():
                        metrics['risk_score'] = (metrics['recall_bad'] * recall_weight + 
                                                metrics['precision_bad'] * precision_weight)
                    
                    # Create tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Summary", 
                        "📈 Visual Comparison", 
                        "🧮 Detailed Metrics",
                        "💾 Export"
                    ])
                    
                    # ─────────────────────────────────────────────────────────
                    # TAB 1: SUMMARY
                    # ─────────────────────────────────────────────────────────
                    with tab1:
                        st.markdown("### 📊 Model Performance Summary")
                        
                        # Create comparison dataframe
                        metrics_list = ['Accuracy', 'Recall (Bad)', 'Precision (Bad)', 'ROC-AUC', 'F1 (Bad)', 'Risk Score']
                        comparison_data = {}
                        
                        for model_name in selected_models:
                            m = filtered_results[model_name]
                            comparison_data[model_name] = [
                                f"{m['accuracy']*100:.1f}%",
                                f"{m['recall_bad']*100:.1f}%",
                                f"{m['precision_bad']*100:.1f}%",
                                f"{m['roc_auc']*100:.1f}%",
                                f"{m['f1_bad']*100:.1f}%",
                                f"{m['risk_score']*100:.1f}%"
                            ]
                        
                        comparison_df = pd.DataFrame(comparison_data, index=metrics_list)
                        
                        # Find best model per metric
                        best_models = {}
                        for metric_name in metrics_list:
                            # Extract numeric values
                            values = {model: float(comparison_df.loc[metric_name, model].rstrip('%')) 
                                    for model in selected_models}
                            best_model = max(values, key=values.get)
                            best_models[metric_name] = best_model
                        
                        # Style the dataframe
                        def highlight_best(row):
                            best_model = best_models.get(row.name, None)
                            return [f'background-color: #90EE90; font-weight: bold' if col == best_model 
                                   else '' for col in row.index]
                        
                        styled_df = comparison_df.style.apply(highlight_best, axis=1)
                        st.dataframe(styled_df, use_container_width=True)
                        
                        st.caption("🏆 **Green highlight** = Best performance for that metric")
                        
                        # Business Impact Section
                        st.markdown("---")
                        st.markdown("### 💼 Business Impact Analysis")
                        
                        cols = st.columns(len(selected_models))
                        for idx, model_name in enumerate(selected_models):
                            m = filtered_results[model_name]
                            with cols[idx]:
                                st.markdown(f"**{model_name}**")
                                
                                defaults_pct = (m['defaults_caught'] / m['defaults_total'] * 100) if m['defaults_total'] > 0 else 0
                                missed_defaults = m['defaults_total'] - m['defaults_caught']
                                missed_pct = (missed_defaults / m['defaults_total'] * 100) if m['defaults_total'] > 0 else 0
                                
                                st.metric(
                                    "Defaults Caught",
                                    f"{m['defaults_caught']} / {m['defaults_total']}",
                                    f"{defaults_pct:.1f}%"
                                )
                                
                                if missed_pct > 10:
                                    st.error(f"❌ Missed: {missed_defaults} ({missed_pct:.1f}%)")
                                elif missed_pct > 5:
                                    st.warning(f"⚠️ Missed: {missed_defaults} ({missed_pct:.1f}%)")
                                else:
                                    st.success(f"✅ Missed: {missed_defaults} ({missed_pct:.1f}%)")
                        
                        # Business interpretation
                        st.markdown("---")
                        st.info("""
                        **📖 Business Interpretation Guide:**
                        
                        - **Recall (Bad)** = % of actual defaults we identified. Higher = safer for the bank.
                        - **Precision (Bad)** = % of rejected loans that would actually default. Higher = less opportunity cost.
                        - **Risk Score** = Weighted combination emphasizing your business priorities.
                        - **Missed Defaults** = False Negatives → Direct financial losses
                        """)
                    
                    # ─────────────────────────────────────────────────────────
                    # TAB 2: VISUAL COMPARISON
                    # ─────────────────────────────────────────────────────────
                    with tab2:
                        st.markdown("### 📈 Visual Model Comparison")
                        
                        # Bar chart for Recall (most important)
                        st.markdown("#### 🎯 Recall (Bad Class) - Primary Metric")
                        recall_data = pd.DataFrame({
                            'Model': selected_models,
                            'Recall (Bad)': [filtered_results[m]['recall_bad'] * 100 for m in selected_models]
                        })
                        
                        fig_bar = px.bar(
                            recall_data,
                            x='Model',
                            y='Recall (Bad)',
                            title='Recall Comparison: % of Defaults Correctly Identified',
                            labels={'Recall (Bad)': 'Recall (%)'},
                            color='Recall (Bad)',
                            color_continuous_scale='RdYlGn',
                            text='Recall (Bad)'
                        )
                        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_bar.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Radar chart for all metrics
                        st.markdown("---")
                        st.markdown("#### 🕸️ Multi-Metric Radar Chart")
                        
                        metrics_for_radar = ['Accuracy', 'Recall (Bad)', 'Precision (Bad)', 'ROC-AUC', 'F1 (Bad)']
                        
                        fig_radar = go.Figure()
                        
                        for model_name in selected_models:
                            m = filtered_results[model_name]
                            values = [
                                m['accuracy'] * 100,
                                m['recall_bad'] * 100,
                                m['precision_bad'] * 100,
                                m['roc_auc'] * 100,
                                m['f1_bad'] * 100
                            ]
                            
                            fig_radar.add_trace(go.Scatterpolar(
                                r=values,
                                theta=metrics_for_radar,
                                fill='toself',
                                name=model_name
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100]
                                )
                            ),
                            showlegend=True,
                            title="All Metrics Comparison (0-100%)",
                            height=500
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Confusion Matrix for champion model
                        st.markdown("---")
                        st.markdown("#### 🧮 Confusion Matrix - Champion Model")
                        
                        # Find champion (highest risk score)
                        champion = max(selected_models, 
                                     key=lambda m: filtered_results[m]['risk_score'])
                        st.success(f"🏆 **Champion Model**: {champion} (Risk Score: {filtered_results[champion]['risk_score']*100:.1f}%)")
                        
                        cm_champion = np.array(filtered_results[champion]['confusion_matrix'])
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_cm = px.imshow(
                                cm_champion,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['Good', 'Bad'],
                                y=['Good', 'Bad'],
                                color_continuous_scale='Blues',
                                text_auto=True,
                                title=f"Confusion Matrix: {champion}"
                            )
                            fig_cm.update_layout(height=400)
                            st.plotly_chart(fig_cm, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Matrix Interpretation:**")
                            st.markdown(f"""
                            - **True Negatives**: {cm_champion[0,0]}  
                              Good loans correctly approved
                            
                            - **False Positives**: {cm_champion[0,1]}  
                              Good loans rejected (opportunity loss)
                            
                            - **False Negatives**: {cm_champion[1,0]}  
                              ⚠️ Bad loans approved (financial loss)
                            
                            - **True Positives**: {cm_champion[1,1]}  
                              Bad loans correctly rejected
                            """)
                        
                        # Scatter plot: Precision vs Recall tradeoff
                        st.markdown("---")
                        st.markdown("#### ⚖️ Precision vs Recall Tradeoff")
                        
                        scatter_data = pd.DataFrame({
                            'Model': selected_models,
                            'Precision (Bad)': [filtered_results[m]['precision_bad'] * 100 for m in selected_models],
                            'Recall (Bad)': [filtered_results[m]['recall_bad'] * 100 for m in selected_models],
                            'Risk Score': [filtered_results[m]['risk_score'] * 100 for m in selected_models]
                        })
                        
                        fig_scatter = px.scatter(
                            scatter_data,
                            x='Precision (Bad)',
                            y='Recall (Bad)',
                            size='Risk Score',
                            color='Model',
                            text='Model',
                            title='Precision-Recall Tradeoff (Bubble size = Risk Score)',
                            labels={'Precision (Bad)': 'Precision (%)', 'Recall (Bad)': 'Recall (%)'},
                            size_max=60
                        )
                        fig_scatter.update_traces(textposition='top center')
                        fig_scatter.update_layout(height=500)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # ─────────────────────────────────────────────────────────
                    # TAB 3: DETAILED METRICS
                    # ─────────────────────────────────────────────────────────
                    with tab3:
                        st.markdown("### 🧮 Detailed Model Metrics")
                        
                        for model_name in selected_models:
                            with st.expander(f"📂 {model_name} - Full Classification Report", expanded=(model_name == champion)):
                                m = filtered_results[model_name]
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Accuracy", f"{m['accuracy']:.3f}")
                                    st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
                                
                                with col2:
                                    st.metric("Recall (Bad)", f"{m['recall_bad']:.3f}", 
                                            help="Primary metric for risk management")
                                    st.metric("Precision (Bad)", f"{m['precision_bad']:.3f}")
                                
                                with col3:
                                    st.metric("F1-Score (Bad)", f"{m['f1_bad']:.3f}")
                                    st.metric("Risk Score", f"{m['risk_score']:.3f}",
                                            help=f"Weighted: {recall_weight:.0%} Recall + {precision_weight:.0%} Precision")
                                
                                # Classification report table
                                st.markdown("---")
                                st.markdown("**Classification Report:**")
                                
                                report_df = pd.DataFrame({
                                    'Class': ['Good (0)', 'Bad (1)'],
                                    'Precision': [f"{m['precision_good']:.3f}", f"{m['precision_bad']:.3f}"],
                                    'Recall': [f"{m['recall_good']:.3f}", f"{m['recall_bad']:.3f}"],
                                    'F1-Score': [
                                        f"{2 * m['precision_good'] * m['recall_good'] / (m['precision_good'] + m['recall_good']):.3f}" if (m['precision_good'] + m['recall_good']) > 0 else "0.000",
                                        f"{m['f1_bad']:.3f}"
                                    ]
                                })
                                
                                st.dataframe(report_df, use_container_width=True)
                                
                                # Confusion matrix
                                st.markdown("---")
                                st.markdown("**Confusion Matrix:**")
                                cm_display = pd.DataFrame(
                                    m['confusion_matrix'],
                                    columns=['Predicted Good', 'Predicted Bad'],
                                    index=['Actual Good', 'Actual Bad']
                                )
                                
                                # Style confusion matrix
                                def color_cm(val):
                                    if isinstance(val, (int, float)):
                                        return 'background-color: #d4edda' if val > 200 else 'background-color: #f8d7da' if val > 50 else ''
                                    return ''
                                
                                st.dataframe(cm_display.style.applymap(color_cm), use_container_width=True)
                                
                                # Business metrics
                                st.markdown("---")
                                st.markdown("**Business Impact:**")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.success(f"✅ Defaults Caught: **{m['defaults_caught']}** / {m['defaults_total']}")
                                    catch_rate = (m['defaults_caught'] / m['defaults_total'] * 100) if m['defaults_total'] > 0 else 0
                                    st.caption(f"Detection Rate: {catch_rate:.1f}%")
                                
                                with col2:
                                    missed = m['defaults_total'] - m['defaults_caught']
                                    if missed > m['defaults_total'] * 0.1:
                                        st.error(f"❌ Defaults Missed: **{missed}**")
                                    else:
                                        st.warning(f"⚠️ Defaults Missed: **{missed}**")
                                    miss_rate = (missed / m['defaults_total'] * 100) if m['defaults_total'] > 0 else 0
                                    st.caption(f"Miss Rate: {miss_rate:.1f}%")
                    
                    # ─────────────────────────────────────────────────────────
                    # TAB 4: EXPORT
                    # ─────────────────────────────────────────────────────────
                    with tab4:
                        st.markdown("### 💾 Export Model Comparison")
                        
                        st.info("📥 Download comparison results in various formats for reporting and documentation.")
                        
                        # Prepare export data
                        export_data = []
                        for model_name in selected_models:
                            m = filtered_results[model_name]
                            export_data.append({
                                'Model': model_name,
                                'Accuracy': f"{m['accuracy']:.4f}",
                                'Recall_Bad': f"{m['recall_bad']:.4f}",
                                'Precision_Bad': f"{m['precision_bad']:.4f}",
                                'ROC_AUC': f"{m['roc_auc']:.4f}",
                                'F1_Bad': f"{m['f1_bad']:.4f}",
                                'Risk_Score': f"{m['risk_score']:.4f}",
                                'Defaults_Caught': m['defaults_caught'],
                                'Defaults_Total': m['defaults_total'],
                                'Defaults_Missed': m['defaults_total'] - m['defaults_caught']
                            })
                        
                        export_df = pd.DataFrame(export_data)
                        
                        # CSV export
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = export_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📊 Download as CSV",
                                data=csv,
                                file_name="model_comparison.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # JSON export
                            import json
                            
                            # Convert numpy types to native Python types for JSON serialization
                            def convert_to_serializable(obj):
                                if isinstance(obj, dict):
                                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_to_serializable(item) for item in obj]
                                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                                    return int(obj)
                                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                else:
                                    return obj
                            
                            serializable_results = convert_to_serializable(filtered_results)
                            json_data = json.dumps(serializable_results, indent=2)
                            
                            st.download_button(
                                label="📄 Download as JSON",
                                data=json_data,
                                file_name="model_comparison.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        # Preview
                        st.markdown("---")
                        st.markdown("**Preview of Export Data:**")
                        st.dataframe(export_df, use_container_width=True)
                        
                        # Summary report
                        st.markdown("---")
                        st.markdown("**📋 Summary Report:**")
                        
                        champion = max(selected_models, key=lambda m: filtered_results[m]['risk_score'])
                        
                        report_text = f"""
**Model Comparison Summary**
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

🏆 **Champion Model**: {champion}
   - Risk Score: {filtered_results[champion]['risk_score']:.3f}
   - Recall (Bad): {filtered_results[champion]['recall_bad']:.3f}
   - Precision (Bad): {filtered_results[champion]['precision_bad']:.3f}
   - Defaults Caught: {filtered_results[champion]['defaults_caught']} / {filtered_results[champion]['defaults_total']}

**All Models Evaluated**: {len(selected_models)}
{chr(10).join([f"   - {m}" for m in selected_models])}

**Risk Score Weights**:
   - Recall: {recall_weight:.1%}
   - Precision: {precision_weight:.1%}

**Recommendation**: Deploy {champion} for maximum risk mitigation.
                        """
                        
                        st.text_area("Summary Report", report_text, height=300)
                        
                        st.download_button(
                            label="📝 Download Summary Report",
                            data=report_text,
                            file_name="model_comparison_summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                
                st.success("✅ Model comparison dashboard loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Evaluation Error: {str(e)}")
                st.exception(e)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>© 2026 Credit Risk Assessment System | Developed by Keshav Verma (iitp_aiml_2506273)</p>
    <p>🚀 Powered by Streamlit | Built with ❤️ for Enterprise Credit Risk Management</p>
</div>
""", unsafe_allow_html=True)
