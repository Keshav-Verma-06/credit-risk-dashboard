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
    """Auto-load model from Models folder"""
    model_path = 'Models/xgboost_model.pkl'
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
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
# PAGE 5: MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Model Performance":
    st.markdown('<div class="main-header">⚙️ Model Performance Metrics</div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.error("❌ Model not loaded. Check that Models/xgboost_model.pkl exists.")
    elif st.session_state.data is None or 'Risk' not in st.session_state.data.columns:
        st.warning("⚠️ Dataset with Risk labels required. Check data/german_credit_data.csv")
    else:
        st.markdown("### 🔬 Evaluate Model on Current Dataset")
        
        if st.button("🚀 Run Model Evaluation", type="primary"):
            try:
                with st.spinner("Evaluating model performance..."):
                    df = st.session_state.data
                    
                    # Preprocess data
                    df_processed = preprocess_data(df)
                    
                    # Separate features and target
                    X = df_processed.drop('Risk', axis=1)
                    y = df_processed['Risk']
                    
                    # Make predictions
                    y_pred = st.session_state.model.predict(X)
                    
                    if hasattr(st.session_state.model, 'predict_proba'):
                        y_pred_proba = st.session_state.model.predict_proba(X)[:, 1]
                    else:
                        y_pred_proba = None
                    
                    # Calculate metrics
                    metrics = calculate_risk_metrics(y, y_pred, y_pred_proba)
                
                st.success("✅ Evaluation completed!")
                
                # Display metrics
                st.markdown("---")
                st.markdown("### 📊 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Recall (Bad)", f"{metrics['recall_bad']:.3f}", 
                             help="Critical metric: % of defaults correctly identified")
                with col3:
                    st.metric("Precision (Bad)", f"{metrics['precision_bad']:.3f}")
                with col4:
                    if 'roc_auc' in metrics:
                        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                
                # Confusion Matrix
                st.markdown("---")
                st.markdown("### 🧮 Confusion Matrix")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    cm = np.array(metrics['confusion_matrix'])
                    
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Good', 'Bad'],
                        y=['Good', 'Bad'],
                        color_continuous_scale='Blues',
                        text_auto=True,
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Business Interpretation")
                    
                    if 'defaults_caught' in metrics:
                        st.success(f"✅ Defaults Caught: {metrics['defaults_caught']} / {metrics['total_defaults']}")
                        st.error(f"⚠️ Defaults Missed: {metrics['defaults_missed']}")
                    
                    st.markdown(f"""
                    **Key Metrics:**
                    - **Recall (Bad)**: {metrics['recall_bad']:.1%}
                      - What % of actual defaults we identified
                      - Higher is better for risk management
                    
                    - **Precision (Bad)**: {metrics['precision_bad']:.1%}
                      - Of loans we reject, how many would actually default
                      - Balance with business opportunity cost
                    """)
                
                # Detailed metrics
                st.markdown("---")
                st.markdown("### 📋 Detailed Metrics")
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision (Good)', 'Precision (Bad)', 
                              'Recall (Good)', 'Recall (Bad)', 'F1-Score'],
                    'Value': [
                        f"{metrics['accuracy']:.4f}",
                        f"{metrics['precision_good']:.4f}",
                        f"{metrics['precision_bad']:.4f}",
                        f"{metrics['recall_good']:.4f}",
                        f"{metrics['recall_bad']:.4f}",
                        f"{metrics['f1_score']:.4f}"
                    ]
                })
                
                st.dataframe(metrics_df, use_container_width=True)
                
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
