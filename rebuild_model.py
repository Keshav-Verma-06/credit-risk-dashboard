"""
Rebuild model on Streamlit Cloud startup if model file is missing
Uses joblib for better sklearn compatibility across versions
"""
import os
import joblib
import pandas as pd
import sys
import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import sklearn

def rebuild_model_if_missing():
    """Train and save model if it doesn't exist"""
    model_path = 'Models/xgboost_model.pkl'
    
    # If model already exists, don't rebuild
    if os.path.exists(model_path):
        return True
    
    print("Model file not found. Rebuilding model...")
    print(f"Environment: Python {sys.version}, sklearn {sklearn.__version__}")
    
    # Load data
    data_path = 'data/german_credit_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found!")
        return False
    
    df = pd.read_csv(data_path)
    
    # Feature engineering
    df['Credit_to_Duration'] = df['Credit amount'] / (df['Duration'] + 1)
    
    # Prepare features and target
    feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 
                   'Checking account', 'Credit amount', 'Duration', 
                   'Purpose', 'Credit_to_Duration']
    
    X = df[feature_cols]
    y = df['Risk'].map({'good': 0, 'bad': 1})  # Try lowercase
    if y.isna().any():
        y = df['Risk'].map({'Good': 0, 'Bad': 1})  # Try capitalized
    
    # Define preprocessing
    numeric_features = ['Age', 'Credit amount', 'Duration', 'Credit_to_Duration']
    categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts', 
                           'Checking account', 'Purpose']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
    
    # Create model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    # Train model
    print("Training model...")
    model.fit(X, y)
    
    # Create Models directory if it doesn't exist
    os.makedirs('Models', exist_ok=True)
    
    # Save model with metadata for version compatibility checking
    model_metadata = {
        'model': model,
        'sklearn_version': sklearn.__version__,
        'trained_at': datetime.datetime.now().isoformat(),
        'python_version': sys.version,
        'feature_columns': feature_cols
    }
    
    # Use joblib.dump instead of pickle for better sklearn compatibility
    joblib.dump(model_metadata, model_path, compress=3)
    
    print(f"Model saved to {model_path}")
    print(f"sklearn version: {sklearn.__version__}")
    return True

if __name__ == "__main__":
    rebuild_model_if_missing()
