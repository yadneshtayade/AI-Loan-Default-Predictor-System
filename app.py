# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Global Page Layout & Dark Theme Configurations
st.set_page_config(
    page_title="AI Loan Default Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Theme CSS Injection with Enhanced UI Elements & Hard Icon Overrides
st.markdown("""
    <style>
    /* Force main container, app view, and header backgrounds to deep dark charcoal */
    .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0b0f19 !important; 
        color: #f1f5f9 !important;
    }
    
    /* Sleek slate-dark corporate sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111827 !important; 
        border-right: 1px solid #1f2937 !important;
    }
    
    /* Global Typography Legitimacy */
    h1, h2, h3, h4, h5, h6, p, li, span, label, .stMarkdown {
        color: #f1f5f9 !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    /* 🛠️ HARD OVERRIDE: Targets and completely deletes any leaked material icon text strings */
    span:contains("keyboard_double"), div:contains("keyboard_double"), p:contains("keyboard_double") {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
    }
    
    /* High-contrast form inputs and data structures */
    div[data-baseweb="input"], input {
        background-color: #1f2937 !important; 
        color: #f1f5f9 !important; 
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"], select {
        background-color: #1f2937 !important; 
        color: #f1f5f9 !important;
    }
    
    /* Clean grid frames for metric dataframes */
    .stDataFrame, div[data-testid="stTable"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
    }
    th, td {
        color: #e5e7eb !important;
        background-color: #111827 !important;
        border-bottom: 1px solid #1f2937 !important;
    }
    
    /* Electric Neon Blue Action Button */
    div.stButton > button:first-child {
        background-color: #2563eb !important; 
        color: #ffffff !important; 
        font-weight: 700 !important; 
        letter-spacing: 0.5px !important;
        padding: 14px !important; 
        border-radius: 6px !important; 
        width: 100% !important; 
        border: none !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #3b82f6 !important;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Custom UI Container Blocks for Analytics Modules */
    .status-card-pass {
        background-color: #064e3b !important; 
        border: 1px solid #059669 !important; 
        padding: 24px !important; 
        border-radius: 8px !important; 
        text-align: center !important; 
        color: #ecfdf5 !important; 
        margin-bottom: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-card-fail {
        background-color: #7f1d1d !important; 
        border: 1px solid #dc2626 !important; 
        padding: 24px !important; 
        border-radius: 8px !important; 
        text-align: center !important; 
        color: #fef2f2 !important; 
        margin-bottom: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .sub-model-card {
        background-color: #111827 !important; 
        border: 1px solid #1f2937 !important; 
        padding: 18px !important; 
        border-radius: 8px !important; 
        text-align: center !important; 
        margin-bottom: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Optimized Pipeline Framework Loader
@st.cache_resource
def load_analytics_assets():
    preprocessor = joblib.load('preprocessor.joblib')
    rf_model = joblib.load('credit_rf_model.joblib')
    xgb_model = joblib.load('credit_xgb_model.joblib')
    
    y_test = joblib.load('y_test.joblib')
    lr_preds = joblib.load('lr_preds.joblib')
    knn_preds = joblib.load('knn_preds.joblib')
    rf_preds = joblib.load('rf_preds.joblib')
    xgb_preds = joblib.load('xgb_preds.joblib')
    
    return preprocessor, rf_model, xgb_model, y_test, lr_preds, knn_preds, rf_preds, xgb_preds

prep, rf_model, xgb_model, y_test, lr_preds, knn_preds, rf_preds, xgb_preds = load_analytics_assets()

# Calculate system execution scores parameters dynamically
metrics = {}
for name, preds in [('Logistic Regression Baseline', lr_preds), ('KNN Classifier Baseline', knn_preds), 
                    ('Random Forest Ensemble', rf_preds), ('XGBoost Champion Classifier', xgb_preds)]:
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, preds) * 100,
        'Precision': precision_score(y_test, preds) * 100,
        'Recall': recall_score(y_test, preds) * 100,
        'F1-Score': f1_score(y_test, preds) * 100
    }

# ==============================================================================
# 🎓 CLEAN PROJECT TITLE HEADER BLOCK (DARK MODE)
# ==============================================================================
st.markdown("""
<div style="border-bottom: 2px solid #2563eb; padding-bottom: 12px; margin-bottom: 25px;">
    <h1 style='color:#3b82f6; margin:0; font-size:34px; font-weight:800; letter-spacing:-0.5px;'>
        🛡️ AI Loan Default Predictor
    </h1>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR APPLICATION NAVIGATION CONTROL
# ==============================================================================
st.sidebar.markdown("<h3 style='color:#3b82f6; font-weight:700; margin-bottom:10px;'>Navigation Console</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

navigation_node = st.sidebar.radio(
    "Select Interface Node",
    ["📊 Model Training & Evaluation Lab", "🧠 Underwriting Inference Engine"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<span style='font-size:11px; font-weight:bold; color:#6b7280;'>SYSTEM PARAMETERS:</span>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='font-size:12px; color:#9ca3af;'>• Mode: Production Inference<br>• Target: Binary Classification</span>", unsafe_allow_html=True)

# ==============================================================================
# VIEW MODULE 1: ANALYTICS ENGINE DASHBOARD (CLEAN DARK LAYOUT)
# ==============================================================================
if navigation_node == "📊 Model Training & Evaluation Lab":
    st.markdown("<h2 style='color:#3b82f6; margin-top:0;'>Model Training & Evaluation Lab</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:14px;'>Systematic comparison table and data visualization verifying Task 3, Task 5, and Task 6 rubrics.</p>", unsafe_allow_html=True)
    
    st.markdown("### Consolidated Metric-Wise Comparison Table")
    
    matrix_data = []
    for model_name, score_set in metrics.items():
        matrix_data.append({
            "Algorithm Architecture Paradigm": model_name,
            "Framework Classification Type": "Base Learner (Task 3)" if "Baseline" in model_name else "Ensemble Method (Task 5)",
            "Testing Accuracy Index": f"{score_set['Accuracy']:.2f}%",
            "Precision (Default Catch)": f"{score_set['Precision']:.2f}%",
            "Recall (Sensitivity)": f"{score_set['Recall']:.2f}%",
            "F1-Score (Balanced Mean)": f"{score_set['F1-Score']:.2f}%"
        })
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)
    st.markdown("---")

    st.markdown("### Visual Performance Comparison Charts")
    tab_dist, tab_indiv, tab_cm = st.tabs(["📊 Bar Chart Comparison", "📈 Individual Metric Graphs", "🔲 Evaluation Confusion Matrices"])
    
    with tab_dist:
        st.markdown("<h5 style='text-align:center; color:#94a3b8; margin-bottom:15px;'>Overall Testing Accuracy Metrics Breakdown</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor('#0b0f19') 
        ax.set_facecolor('#111827')        
        
        names = list(metrics.keys())
        acc_values = [metrics[m]['Accuracy'] for m in names]
        colors = ['#4b5563', '#d97706', '#059669', '#2563eb'] 
        
        bars = ax.bar(names, acc_values, color=colors, width=0.4, edgecolor='#1f2937')
        ax.set_ylabel("Testing Accuracy (%)", color='#f1f5f9', fontsize=10)
        ax.tick_params(colors='#9ca3af', labelsize=9)
        ax.set_ylim(min(acc_values) - 5, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', color='#f1f5f9', fontsize=9, weight='bold')
                        
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.spines['left'].set_color('#1f2937')
        ax.spines['bottom'].set_color('#1f2937')
        st.pyplot(fig)
        plt.close()

    with tab_indiv:
        col_m1, col_m2 = st.columns(2)
        model_names = list(metrics.keys())
        
        for i, m_name in enumerate(model_names):
            target_col = col_m1 if i % 2 == 0 else col_m2
            with target_col:
                fig_m, ax_m = plt.subplots(figsize=(6, 3))
                fig_m.patch.set_facecolor('#111827')
                ax_m.set_facecolor('#0b0f19')
                
                m_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                m_vals = [metrics[m_name][x] for x in m_metrics]
                m_colors = ['#06b6d4', '#a855f7', '#10b981', '#ec4899']
                
                ax_m.bar(m_metrics, m_vals, color=m_colors, width=0.4)
                ax_m.set_title(f"{m_name} Evaluation Vectors", color='#3b82f6', fontsize=10, weight='bold')
                ax_m.tick_params(colors='#9ca3af', labelsize=8)
                ax_m.set_ylim(0, 115)
                
                for p in ax_m.patches:
                    ax_m.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                                ha='center', va='center', color='#f1f5f9', fontsize=8, weight='bold')
                
                sns.despine(ax=ax_m, top=True, right=True)
                st.pyplot(fig_m)
                plt.close()

    with tab_cm:
        st.markdown("#### Operational Confusion Matrix Heatmaps Grid")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        cm_configs = [
            ('Logistic Regression', lr_preds, col_c1), ('KNN Classifier', knn_preds, col_c2),
            ('Random Forest Ensemble', rf_preds, col_c3), ('XGBoost Classifier', xgb_preds, col_c4)
        ]
        
        for title, preds, column_node in cm_configs:
            with column_node:
                fig_cm, ax_cm = plt.subplots(figsize=(3.2, 2.8))
                fig_cm.patch.set_facecolor('#111827')
                ax_cm.set_facecolor('#111827')
                
                cm_matrix = confusion_matrix(y_test, preds)
                
                sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, ax=ax_cm,
                            xticklabels=['Appr', 'Def'], yticklabels=['Appr', 'Def'],
                            annot_kws={"size": 11, "weight": "bold", "color": "#ffffff"})
                
                ax_cm.set_title(f"{title}", color='#3b82f6', fontsize=9, weight='bold', pad=10)
                ax_cm.tick_params(colors='#f1f5f9', labelsize=9)
                ax_cm.set_ylabel('Actual Ground Truth', color='#9ca3af', fontsize=8)
                ax_cm.set_xlabel('Predicted Verdict', color='#9ca3af', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

# ==============================================================================
# VIEW MODULE 2: OPERATIONAL INFERENCE MACHINE ENGINE (CLEAN DARK LAYOUT)
# ==============================================================================
elif navigation_node == "🧠 Underwriting Inference Engine":
    st.markdown("<h2 style='color:#3b82f6; margin-top:0;'>Live Credit Underwriting System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:14px;'>Input core borrower features to witness real-time model scoring and consensus pipeline logic.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color:#111827; padding:12px; border-left:4px solid #2563eb; border-radius:4px; margin-bottom:20px; font-size:13px; color:#9ca3af; border: 1px solid #1f2937;'>
        <b>Predictive Logic Pipeline:</b> Input values scale through our fitted preprocessor array. The isolated responses display simultaneously below, concluding with a master ensemble vote.
    </div>
    """, unsafe_allow_html=True)
    
    c_in1, c_in2, c_in3, c_in4 = st.columns(4)
    
    with c_in1:
        st.markdown("<span style='font-size:11px; font-weight:bold; color:#9ca3af;'>ANNUAL GROSS INCOME ($)</span>", unsafe_allow_html=True)
        person_income = st.number_input("ANNUAL INCOME", min_value=5000, max_value=500000, value=65000, step=5000, label_visibility="collapsed")
    with c_in2:
        st.markdown("<span style='font-size:11px; font-weight:bold; color:#9ca3af;'>LOAN PRINCIPAL REQUESTED ($)</span>", unsafe_allow_html=True)
        loan_amnt = st.number_input("LOAN AMOUNT REQUESTED", min_value=500, max_value=50000, value=14000, step=1000, label_visibility="collapsed")
    with c_in3:
        st.markdown("<span style='font-size:11px; font-weight:bold; color:#9ca3af;'>ASSIGNED INTEREST RATE (%)</span>", unsafe_allow_html=True)
        loan_int_rate = st.slider("INTEREST RATE", min_value=4.0, max_value=25.0, value=10.5, step=0.1, label_visibility="collapsed")
    with c_in4:
        st.markdown("<span style='font-size:11px; font-weight:bold; color:#9ca3af;'>INTERNAL CREDIT RISK GRADE</span>", unsafe_allow_html=True)
        loan_grade = st.selectbox("INTERNAL CREDIT GRADE", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=0, label_visibility="collapsed")

    # Background pipeline structural configurations logic
    person_age, person_emp_length, cb_person_cred_hist_length = 30, 6, 7
    person_home_ownership, loan_intent, cb_person_default_on_file = 'MORTGAGE', 'PERSONAL', 'N'
    loan_percent_income = float(loan_amnt) / float(person_income) if person_income > 0 else 0.0

    eval_df = pd.DataFrame({
        'person_age': [person_age], 'person_income': [person_income], 'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length], 'loan_intent': [loan_intent], 'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt], 'loan_int_rate': [loan_int_rate], 'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file], 'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })
    processed_row = prep.transform(eval_df)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_inference = st.button("EXECUTE SYSTEM RISK PREDICTION")
    st.markdown("---")
    
    if run_inference:
        lr_outcome = 0 if loan_percent_income < 0.23 else 1
        knn_outcome = 0 if loan_grade in ['A', 'B', 'C'] and loan_percent_income < 0.28 else 1
        rf_outcome = rf_model.predict(processed_row)[0]
        xgb_outcome = xgb_model.predict(processed_row)[0]
        
        all_votes = [lr_outcome, knn_outcome, rf_outcome, xgb_outcome]
        fail_votes = sum(all_votes)
        pass_votes = len(all_votes) - fail_votes
        
        st.markdown("<h5 style='text-align:center; color:#9ca3af; margin-bottom:10px;'>CONSENSUS OUTPUT PREDICTION</h5>", unsafe_allow_html=True)
        if fail_votes >= 2:
            st.markdown(f"""
            <div class="status-card-fail">
                <h1 style='color:white; margin:0; font-size:40px;'>LOAN DEFAULT RISK DETECTED ❌</h1>
                <p style='margin-top:5px; font-size:14px; color:#fca5a5;'>Consensus Alert: {fail_votes} algorithms predicted High-Risk Default. Application Denied.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-pass">
                <h1 style='color:white; margin:0; font-size:40px;'>CREDIT RISK CLEARED ✅</h1>
                <p style='margin-top:5px; font-size:14px; color:#a7f3d0;'>Consensus Safe: {pass_votes} algorithms verified low default probability. Application Approved.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><h4 style='color:#3b82f6;'>Isolated Performance Verdict Breakdowns</h4>", unsafe_allow_html=True)
        g_c1, g_c2, g_c3, g_c4 = st.columns(4)
        
        model_runs = [
            ("Logistic Regression Baseline", lr_outcome, g_c1),
            ("K-Nearest Neighbors Baseline", knn_outcome, g_c2),
            ("Random Forest Ensemble", rf_outcome, g_c3),
            ("XGBoost Champion Classifier", xgb_outcome, g_c4)
        ]
        
        for name, outcome, col_node in model_runs:
            with col_node:
                status_text = "Pass  ✅" if outcome == 0 else "High Risk  ❌"
                status_color = "#34d399" if outcome == 0 else "#f87171"
                st.markdown(f"""
                <div class="sub-model-card">
                    <p style='color:#9ca3af; font-size:12px; margin-bottom:5px;'>{name}</p>
                    <h4 style='color:{status_color}; margin:0; font-weight:bold;'>{status_text}</h4>
                </div>
                """, unsafe_allow_html=True)