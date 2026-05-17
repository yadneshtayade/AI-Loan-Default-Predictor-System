# train_ensemble.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

print("⏳ Loading preprocessed training arrays and base model results...")
# Navigating up one folder level using '../' to grab the baseline files securely
X_train_processed = joblib.load('../X_train_processed.joblib')
X_test_processed = joblib.load('../X_test_processed.joblib')
y_train = joblib.load('../y_train.joblib')
y_test = joblib.load('../y_test.joblib')
lr_preds = joblib.load('../lr_preds.joblib')
knn_preds = joblib.load('../knn_preds.joblib')
preprocessor = joblib.load('../preprocessor.joblib')

# =========================================================
# TASK 5 (k): APPLY AT LEAST 2 ENSEMBLE TECHNIQUES
# =========================================================
print("\n🚀 [TASK 5 - k] Training Ensemble Technique 1: Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_processed, y_train)
rf_preds = rf_model.predict(X_test_processed)

print("🚀 [TASK 5 - k] Training Ensemble Technique 2: XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_processed, y_train)
xgb_preds = xgb_model.predict(X_test_processed)

# Print standard classification breakdown to console
print("\n=================== RANDOM FOREST REPORT ===================")
print(classification_report(y_test, rf_preds))

print("=================== XGBOOST ENSEMBLE REPORT ===================")
print(classification_report(y_test, xgb_preds))

# Save the ensemble models back to the parent directory so the app dashboard can load them
joblib.dump(rf_model, '../credit_rf_model.joblib')
joblib.dump(xgb_model, '../credit_xgb_model.joblib')
joblib.dump(rf_preds, '../rf_preds.joblib')
joblib.dump(xgb_preds, '../xgb_preds.joblib')
print("💾 Both ensemble model binaries archived successfully!")

# =========================================================
# TASK 5 (m): VISUALIZE FEATURE IMPORTANCE FOR TREE MODELS
# =========================================================
print("\n📊 [TASK 5 - m] Extracting and plotting feature importance matrix...")

# Re-extract real column feature names from our preprocessing pipeline engine
num_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
all_features = num_features + encoded_cat_names

# Pull importance rankings from XGBoost architecture weights
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1][:10] # Filter down to top 10 markers

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=np.array(all_features)[indices], palette="viridis")
plt.title("Top 10 Risk Indicators Dictating Credit Defaults (XGBoost Feature Importance)")
plt.xlabel("Relative Statistical Importance Weight")
plt.tight_layout()
plt.savefig('../feature_importance.png') # Export chart image directly to your parent folder for easy slides capture
plt.close()
print("✅ Visual chart successfully exported as 'feature_importance.png'!")

# =========================================================
# TASK 5 (l & n): COMPARISON MATRIX & REPORT ACCURACY TABLES
# =========================================================
print("\n📈 [TASK 5 - l & n] Generating Accuracy Performance Comparison Table...")

# Calculate absolute test accuracy percentages across both paradigms
lr_acc = accuracy_score(y_test, lr_preds) * 100
knn_acc = accuracy_score(y_test, knn_preds) * 100
rf_acc = accuracy_score(y_test, rf_preds) * 100
xgb_acc = accuracy_score(y_test, xgb_preds) * 100

# Build clean summary comparison layout
comparison_table = pd.DataFrame({
    'Model Approach': [
        'Logistic Regression (Base)', 
        'K-Nearest Neighbors (Base)', 
        'Random Forest (Ensemble)', 
        'XGBoost (Ensemble)'
    ],
    'Accuracy Metric (%)': [
        f"{lr_acc:.2f}%", 
        f"{knn_acc:.2f}%", 
        f"{rf_acc:.2f}%", 
        f"{xgb_acc:.2f}%"
    ],
    'Net Improvement over Baseline': [
        'Baseline Benchmark Reference', 
        f"{(knn_acc - lr_acc):+.2f}%", 
        f"{(rf_acc - lr_acc):+.2f}%", 
        f"{(xgb_acc - lr_acc):+.2f}%"
    ]
})

print("\n" + "="*80)
print(comparison_table.to_string(index=False))
print("="*80)
print("\n✅ Task 5 operations complete! Evaluation results printed above.")