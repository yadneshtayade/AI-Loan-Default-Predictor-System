# train_base.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

print("⏳ Step 1: Loading raw credit dataset from disk...")
# Corrected directory path targeting your subfolder structure
df = pd.read_csv('Credit_Risk_Predictor/dataset/credit_risk_dataset.csv')

print("⏳ Step 2: Handling missing values via median imputation...")
# Safely fill missing row values to prevent downstream computation failures
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# Separate features (X) from the target classification column (y)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

print("⏳ Step 3: Splitting dataset into Train (80%) and Test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Group variables for numerical scaling and categorical array layout tracking
num_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

print("⏳ Step 4: Constructing mathematical preprocessing pipeline...")
# ColumnTransformer normalizes numerical fields and encodes textual elements safely
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
    ])

# Fit data configurations and process matrices
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Export assets to disk so neighboring project files can read them
joblib.dump(preprocessor, 'preprocessor.joblib')
print("💾 Preprocessor configuration saved successfully as 'preprocessor.joblib'")

# =========================================================
# TASK 3: IMPLEMENT BASELINE MODELS
# =========================================================
print("\n🚀 [TASK 3] Training Base Model 1: Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_processed, y_train)
lr_preds = lr_model.predict(X_test_processed)

print("🚀 [TASK 3] Training Base Model 2: K-Nearest Neighbors (KNN)...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_processed, y_train)
knn_preds = knn_model.predict(X_test_processed)

# =========================================================
# EVALUATING BASE MODELS
# =========================================================
print("\n================== LOGISTIC REGRESSION REPORT ==================")
print(classification_report(y_test, lr_preds))

print("====================== KNN MODEL REPORT ======================")
print(classification_report(y_test, knn_preds))

# Stash intermediate array results for train_ensemble.py processing
joblib.dump(y_test, 'y_test.joblib')
joblib.dump(lr_preds, 'lr_preds.joblib')
joblib.dump(knn_preds, 'knn_preds.joblib')
joblib.dump(X_train_processed, 'X_train_processed.joblib')
joblib.dump(X_test_processed, 'X_test_processed.joblib')
joblib.dump(y_train, 'y_train.joblib')

print("\n✅ File 1 Execution Complete! Baseline models computed successfully.")