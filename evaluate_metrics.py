# evaluate_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

print("⏳ Loading stored predictions across all evaluated phases...")
y_test = joblib.load('y_test.joblib')
lr_preds = joblib.load('lr_preds.joblib')
knn_preds = joblib.load('knn_preds.joblib')
rf_preds = joblib.load('rf_preds.joblib')
xgb_preds = joblib.load('xgb_preds.joblib')

# Compute testing accuracies across all four models
lr_acc = accuracy_score(y_test, lr_preds) * 100
knn_acc = accuracy_score(y_test, knn_preds) * 100
rf_acc = accuracy_score(y_test, rf_preds) * 100
xgb_acc = accuracy_score(y_test, xgb_preds) * 100

# =========================================================
# TASK 5 (l / n) & TASK 6: CONSOLIDATED TABLE
# =========================================================
print("\n📈 [TASK 5 & 6] Generating Consolidated Performance Evaluation Table...")

comparison_df = pd.DataFrame({
    'Model Framework Paradigm': [
        'Logistic Regression (Base)', 
        'K-Nearest Neighbors (Base)', 
        'Random Forest (Ensemble)', 
        'XGBoost (Ensemble)'
    ],
    'Testing Accuracy Score': [
        f"{lr_acc:.2f}%", 
        f"{knn_acc:.2f}%", 
        f"{rf_acc:.2f}%", 
        f"{xgb_acc:.2f}%"
    ],
    'Improvement Over Baseline': [
        'Baseline Benchmark', 
        f"{(knn_acc - lr_acc):+.2f}%", 
        f"{(rf_acc - lr_acc):+.2f}%", 
        f"{(xgb_acc - lr_acc):+.2f}%"
    ]
})

print("\n" + "="*75)
print(comparison_df.to_string(index=False))
print("="*75)

# =========================================================
# TASK 6: BAR CHART VISUAL REVIEWS
# =========================================================
print("\n📊 [TASK 6] Plotting comparative metric charts for final demonstration review...")
models = ['Logistic Reg', 'KNN', 'Random Forest', 'XGBoost']
accuracies = [lr_acc, knn_acc, rf_acc, xgb_acc]

plt.figure(figsize=(8, 4))
ax = sns.barplot(x=models, y=accuracies, palette="pastel")
plt.title("Model Paradigm Accuracy Score Comparisons")
plt.ylabel("Testing Accuracy (%)")
plt.ylim(min(accuracies) - 5, max(accuracies) + 2)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('model_accuracy_comparison.png')
plt.close()

# Generate Confusion Matrix Heatmap for evaluation audit
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Approved (Safe)', 'Denied (Risk)'], yticklabels=['Approved (Safe)', 'Denied (Risk)'])
plt.title("XGBoost Confusion Evaluation Matrix")
plt.ylabel('Actual Status (Ground Truth)')
plt.xlabel('Predicted Status (Model Verdict)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("✅ evaluate_metrics.py Execution Complete! Evaluation reports ready.")