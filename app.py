import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Page Config
st.set_page_config(page_title="Breast Cancer Classification | 2025AA05371", layout="wide")

st.title("🔬 Breast Cancer Prediction App | By - 2025AA05371")
st.markdown("""
This app predicts whether a tumor is **Malignant** (Cancerous: 1) or **Benign** (Non-Cancerous: 0) using Machine Learning.
Upload a CSV file to see predictions and evaluation metrics. (If not uploaded loads default test set data)
""")

# --- Sidebar ---
st.sidebar.header("Configuration")

# 1. Model Selection [Requirement 92]
model_options = [
    "Logistic Regression", 
    "Decision Tree", 
    "KNN", 
    "Naive Bayes", 
    "Random Forest", 
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select ML Model", model_options)

# 2. File Uploader [Requirement 91]
uploaded_file = st.sidebar.file_uploader("Upload your input CSV", type=["csv"])

# --- Helper Functions ---
def load_model(model_name):
    filename = f"model/{model_name.lower().replace(' ', '_')}.pkl"
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def load_scaler():
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# --- Main App Logic ---

# Load Scaler and Model
try:
    scaler = load_scaler()
    model = load_model(selected_model_name)
except FileNotFoundError:
    st.error("Model files not found! Please run 'train_model.py' locally first to generate the models.")
    st.stop()

if uploaded_file is None:
    df = pd.read_csv("test_data.csv")
elif uploaded_file is not None:
    # Read Data
    df = pd.read_csv(uploaded_file)
else:
    st.info("👈 Please upload a CSV file from the sidebar to start.")
    st.markdown("For testing, you can download a sample CSV of the breast cancer test data.")

try:
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Preprocessing checks
    # Assuming the uploaded file has the 'target' column for metrics calculation
    # If dataset is from Kaggle, target might be 'diagnosis' (M/B) or 'target' (0/1)
    
    target_col = None
    if 'target' in df.columns:
        target_col = 'target'
    elif 'diagnosis' in df.columns:
        target_col = 'diagnosis'
        # Convert M/B to 0/1 if necessary (M=0, B=1 in some, check your specific training)
        # In sklearn dataset: 0 = Malignant, 1 = Benign usually, OR vice versa. 
        # Standard sklearn load_breast_cancer: 0 = Malignant, 1 = Benign.
        # However, usually we treat Malignant as the Positive class (1).
        # Let's assume standard numerical input for simplicity in this assignment context.
        pass

    # Split features and target
    if target_col:
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
    else:
        X_test = df
        y_test = None
        st.warning("⚠️ No target column ('target' or 'diagnosis') found. Metrics cannot be calculated, only predictions will be shown.")

    # Ensure features match training (30 features)
    # In a real app, we would explicitly check feature names. 
    # Here we assume the CSV is clean (subset of original data).
    
    # Scale Data
    X_test_scaled = scaler.transform(X_test)

    # Prediction
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Display Results
    st.subheader(f"Results using {selected_model_name}")

    # Metrics [Requirement 93]
    if y_test is not None:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.4f}" if y_prob is not None else "N/A")
        col3.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
        col4.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        col6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        # Confusion Matrix [Requirement 94]
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
    
    # Show Predictions
    st.write("### Raw Predictions")
    results_df = X_test.copy()
    results_df['Predicted_Class'] = y_pred
    if y_prob is not None:
        results_df['Prediction_Probability'] = y_prob
    st.dataframe(results_df)

except Exception as e:
    st.error(f"Error processing the file: {e}")