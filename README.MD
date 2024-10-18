# Credit Card Fraud Detection

## Project Overview
This project aims to build a machine learning model to detect fraudulent credit card transactions. The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle, which contains anonymized features of transactions along with a label indicating whether the transaction is fraudulent or not.

### Dataset
- The dataset contains 284,807 transactions.
- Out of these, only 492 transactions (0.17%) are fraudulent.

### Workflow
1. **Data Preprocessing**:
   - Load the data and split it into training and testing sets.
   - Handle the class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
   
2. **Model Building**:
   - A Random Forest classifier is used to detect fraud.

3. **Model Evaluation**:
   - The model is evaluated using confusion matrix, classification report, and ROC AUC score.

### Files Structure
