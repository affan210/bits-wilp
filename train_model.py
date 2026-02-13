import pandas as pd
import numpy as np
import os
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 1. Create model directory
if not os.path.exists('model'):
    os.makedirs('model')

# 2. Load Dataset (Breast Cancer Wisconsin Diagnostic)
data = pd.read_csv("breast_cancer_wisconsin_data.csv").drop(columns=['Unnamed: 32'], errors='ignore')  # load_breast_cancer()
print(data.columns)
print(data.isna().sum())
data = data.dropna()
print(data.shape)
X = data.drop(columns=['diagnosis', 'id'])  # pd.DataFrame(data.data, columns=data.feature_names)
y = data['diagnosis'].map({'M': 1, 'B': 0})  # Malignant = Cancer := 1 Benign = Non-Cancerous := 0 # pd.Series(data.target)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save Test CSV
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv("test_data.csv", index=False)

# 3. Preprocessing (Scaling is crucial for KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for use in the app
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 4. Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 5. Train and Save Models
results = {}

print("Training models and saving to 'model/' directory...")
for name, model in models.items():
    # Train
    # Tree-based models (RF, DT, XGB) don't strictly need scaling, but it doesn't hurt. 
    # We use scaled data for consistency.
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results[name] = [accuracy, auc, precision, recall, f1, mcc]
    
    # Save Model
    filename = f"model/{name.lower().replace(' ', '_')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name} to {filename}")

# Optional: Print results to console to help with the Observation Table
print("\n--- Model Performance (Test Set) ---")
metrics_df = pd.DataFrame(results, index=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]).T
print(metrics_df)