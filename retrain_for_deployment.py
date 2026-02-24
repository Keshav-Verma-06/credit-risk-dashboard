"""
Retrain Model for Deployment
Creates a new model file compatible with current sklearn version
Use this script when deploying to ensure version compatibility
"""
import os
import sys
import datetime
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import sklearn

print("=" * 60)
print("RETRAINING MODEL FOR DEPLOYMENT")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"sklearn: {sklearn.__version__}")
print(f"Date: {datetime.datetime.now()}")
print("=" * 60)

# Load data
data_path = 'data/german_credit_data.csv'
if not os.path.exists(data_path):
    print(f"❌ ERROR: Data file {data_path} not found!")
    sys.exit(1)

print(f"\n✅ Loading data from {data_path}...")
df = pd.read_csv(data_path)
print(f"   Shape: {df.shape}")

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

print(f"   Features: {len(feature_cols)}")
print(f"   Samples: {len(X)}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

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
print(f"\n🚀 Training XGBoost model...")
model.fit(X, y)
print(f"   ✅ Training complete!")

# Evaluate quickly
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba)
print(f"   Training Accuracy: {acc:.4f}")
print(f"   Training ROC-AUC: {auc:.4f}")

# Create Models directory if it doesn't exist
os.makedirs('Models', exist_ok=True)

# Backup existing model if it exists
model_path = 'Models/xgboost_model.pkl'
if os.path.exists(model_path):
    backup_path = f'Models/xgboost_model_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    print(f"\n📦 Backing up existing model to: {backup_path}")
    os.rename(model_path, backup_path)

# Save model with metadata
model_metadata = {
    'model': model,
    'sklearn_version': sklearn.__version__,
    'trained_at': datetime.datetime.now().isoformat(),
    'python_version': sys.version,
    'feature_columns': feature_cols,
    'training_accuracy': float(acc),
    'training_roc_auc': float(auc),
    'num_features': len(feature_cols),
    'num_samples': len(X)
}

print(f"\n💾 Saving model with metadata...")
joblib.dump(model_metadata, model_path, compress=3)

file_size = os.path.getsize(model_path) / 1024  # KB
print(f"   ✅ Model saved to: {model_path}")
print(f"   File size: {file_size:.2f} KB")

# Verify model loads
print(f"\n🔍 Verifying model can be loaded...")
try:
    loaded_data = joblib.load(model_path)
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        print(f"   ✅ Model loads successfully!")
        print(f"   sklearn version: {loaded_data['sklearn_version']}")
        print(f"   Trained at: {loaded_data['trained_at']}")
    else:
        print(f"   ✅ Model loads (old format)")
except Exception as e:
    print(f"   ❌ ERROR loading model: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ DEPLOYMENT MODEL READY!")
print("=" * 60)
print(f"Model file: {model_path}")
print(f"sklearn version: {sklearn.__version__}")
print(f"Next steps:")
print(f"  1. Commit model file to git")
print(f"  2. Push to GitHub")
print(f"  3. Deploy to Streamlit Cloud")
print(f"  4. Model will load compatible with sklearn {sklearn.__version__}")
print("=" * 60)
