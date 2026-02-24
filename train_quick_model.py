"""
Quick Model Training Script
This script trains a simple XGBoost model using synthetic data or provided data
Use this to quickly generate a model file for testing the dashboard

Run: python train_quick_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle
import os

print("=" * 60)
print("🚀 Quick Model Training Script")
print("=" * 60)

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Check if data exists
data_path = 'data/german_credit_data.csv'

if os.path.exists(data_path):
    print(f"\n✅ Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} records")
else:
    print("\n⚠️  No dataset found. Generating synthetic data...")
    
    # Generate synthetic credit risk data
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 75, n_samples),
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Job': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'Housing': np.random.choice(['own', 'rent', 'free'], n_samples, p=[0.6, 0.3, 0.1]),
        'Saving accounts': np.random.choice(['little', 'moderate', 'quite rich', 'rich', np.nan],
                                           n_samples, p=[0.4, 0.3, 0.1, 0.1, 0.1]),
        'Checking account': np.random.choice(['little', 'moderate', 'rich', np.nan],
                                            n_samples, p=[0.5, 0.25, 0.15, 0.1]),
        'Credit amount': np.random.randint(250, 20000, n_samples),
        'Duration': np.random.randint(4, 72, n_samples),
        'Purpose': np.random.choice(['car', 'furniture', 'radio/TV', 'education', 'business',
                                    'domestic appliances', 'repairs', 'vacation/others'],
                                   n_samples, p=[0.3, 0.15, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05])
    }
    
    # Generate risk based on business rules
    def generate_risk(row):
        risk_score = 0
        if row['Age'] < 25:
            risk_score += 15
        elif row['Age'] > 60:
            risk_score += 10
        if row['Credit amount'] > 10000:
            risk_score += 20
        if row['Duration'] > 36:
            risk_score += 15
        if pd.isna(row['Saving accounts']) or row['Saving accounts'] == 'little':
            risk_score += 20
        if pd.isna(row['Checking account']) or row['Checking account'] == 'little':
            risk_score += 15
        if row['Job'] == 0:
            risk_score += 25
        risk_score += np.random.randint(-10, 10)
        return 1 if risk_score > 50 else 0  # 1 = Bad, 0 = Good
    
    df = pd.DataFrame(data)
    df['Risk'] = df.apply(generate_risk, axis=1)
    
    # Save synthetic data
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"   ✅ Generated and saved {len(df)} synthetic records")

# Preprocessing
print("\n📊 Preprocessing data...")

# Handle missing values
for col in ['Saving accounts', 'Checking account']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Feature Engineering
df['Credit_to_Duration'] = df['Credit amount'] / (df['Duration'] + 1)

age_bins = [17, 25, 35, 50, 100]
age_labels = ['18-25', '26-35', '36-50', '50+']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
df['Age_Group'] = df['Age_Group'].astype(str)

print("   ✅ Feature engineering completed")

# Prepare train-test split
X = df.drop('Risk', axis=1, errors='ignore')
y = df['Risk'] if 'Risk' in df.columns else None

if y is None:
    print("\n❌ Error: No 'Risk' column found in dataset")
    exit(1)

# Encode target variable if it's categorical
if y.dtype == 'object':
    print(f"   Converting Risk labels: Good → 0, Bad → 1")
    y = y.map({'Good': 0, 'Bad': 1})
    if y.isnull().any():
        print(f"   ⚠️  Warning: Unknown Risk values found, filling with 0")
        y = y.fillna(0)

print(f"\n🔀 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# Define preprocessing pipeline
print("\n⚙️  Building preprocessing pipeline...")

numeric_features = ['Age', 'Credit amount', 'Duration', 'Credit_to_Duration']
categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age_Group']

# Filter to existing columns
numeric_features = [f for f in numeric_features if f in X_train.columns]
categorical_features = [f for f in categorical_features if f in X_train.columns]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Create full pipeline with XGBoost
print("\n🤖 Training XGBoost model...")

# Calculate scale_pos_weight for imbalanced data
n_good = (y_train == 0).sum()
n_bad = (y_train == 1).sum()
scale_pos_weight = n_good / n_bad if n_bad > 0 else 1

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    ))
])

# Train model
model_pipeline.fit(X_train, y_train)
print("   ✅ Model training completed")

# Evaluate on test set
print("\n📈 Evaluating model on test set...")
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall_bad = recall_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"   • Accuracy:     {accuracy:.4f}")
print(f"   • Recall (Bad): {recall_bad:.4f}  ← Target metric")
print(f"   • ROC-AUC:      {roc_auc:.4f}")

# Save model
print("\n💾 Saving model...")
os.makedirs('Models', exist_ok=True)
model_path = 'Models/xgboost_model.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model_pipeline, f)

print(f"   ✅ Model saved to: {model_path}")

# Save test data for dashboard evaluation
print("\n💾 Saving test data for dashboard evaluation...")
os.makedirs('Processed_Data', exist_ok=True)

test_data_with_labels = X_test.copy()
test_data_with_labels['Risk'] = y_test.values

test_data_with_labels.to_csv('Processed_Data/test_data.csv', index=False)
print("   ✅ Test data saved to: Processed_Data/test_data.csv")

# Summary
print("\n" + "=" * 60)
print("✅ SUCCESS! Model training completed")
print("=" * 60)
print("\n📦 Generated Files:")
print(f"   1. {model_path}")
print(f"   2. Processed_Data/test_data.csv")
print(f"   3. {data_path}")
print("\n🚀 Next Steps:")
print("   1. Run the dashboard: streamlit run app.py")
print("   2. The model will be auto-loaded from Models/ folder")
print("   3. Use 'Model Performance' page with test_data.csv")
print("\n" + "=" * 60)
