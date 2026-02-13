# bits-wilp
# Breast Cancer Classification App

## 1. Problem Statement
The goal of this project is to develop a machine learning application that can classify breast cancer tumors as either **Malignant** or **Benign** based on diagnostic measurements. Early and accurate diagnosis is critical for effective treatment. This project implements multiple classification algorithms to find the best model for this task.

## 2. Dataset Description
* **Source:** Breast Cancer Wisconsin (Diagnostic) Data Set (available via Kaggle/UCI).
* **Instances:** 569
* **Features:** 30 numeric features (computed from a digitized image of a fine needle aspirate of a breast mass).
* **Target:** Binary classification (Malignant vs Benign).
* **Key Features:** Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, etc.

## 3. Models Used & Performance Comparison
The following models were trained and evaluated on the test set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** |  0.973684  |  0.997380  |  0.976190  |  0.953488  |  0.964706  |  0.943898  |
| **Decision Tree** |  0.947368  |  0.943990  |  0.930233  |  0.930233  |  0.930233  |  0.887979  |
| **KNN** |  0.947368  |  0.981985  |  0.930233  |  0.930233  |  0.930233  |  0.887979  |
| **Naive Bayes** |  0.964912  |  0.997380  |  0.975610  |  0.930233  |  0.952381  |  0.925285  |
| **Random Forest** |  0.964912  |  0.995251  |  0.975610  |  0.930233  |  0.952381  |  0.925285  |
| **XGBoost** |  0.956140  |  0.990829  |  0.952381  |  0.930233  |  0.941176  |  0.906379  |


## 4. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** |  Best overall model with highest accuracy, AUC, and F1-score, showing strong and balanced performance  |
| **Decision Tree** |  Lowest performing model with comparatively lower accuracy and AUC, indicating weaker generalization  |
| **KNN** |  Performs better than Decision Tree with strong AUC but moderate overall classification metrics  |
| **Naive Bayes** |  Very strong AUC and good balance between precision and recall, performing close to the top model  |
| **Random Forest** |  Stable and high-performing ensemble model with strong accuracy and AUC, slightly below Logistic Regression  |
| **XGBoost** |  Good overall performance with balanced metrics but does not outperform the top models in this case  |