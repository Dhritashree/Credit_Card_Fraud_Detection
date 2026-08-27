# Credit Card Fraud Detection

## Project Overview
This project detects fraudulent credit card transactions using Machine Learning and Artificial Neural Network models.

**Problem Type:** Binary Classification  
**Domain:** Finance / Fraud Detection

## Dataset
- Source: Credit Card Fraud Detection Dataset
- Total Transactions: 284,807
- Features: 30 (V1–V28 + Time + Amount)
- Target: Class (0 = Normal, 1 = Fraud)
- Note: Dataset is highly imbalanced
- Dataset download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Models Used
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Artificial Neural Network (ANN)

## Results

| Model               | Accuracy | Precision | Recall  | F1-Score |
|---------------------|----------|-----------|---------|----------|
| Logistic Regression | 0.9991   | 0.8485    | 0.5895  | 0.6957   |
| **Random Forest**   | 0.9995   | 0.9722    | 0.7368  | **0.8383** |
| Gradient Boosting   | 0.9993   | 0.8732    | 0.6526  | 0.7470   |
| ANN                 | 0.9994   | 0.8780    | 0.7579  | 0.8136   |

**Best Model:** Random Forest (Highest F1-Score)

## Key Insights
- Accuracy is not a good metric due to class imbalance
- Random Forest gave the best performance
- Most important features: V17, V14, V12

## How to Run
1. Download the notebook
2. Upload it to Google Colab
3. Upload the creditcard.csv file
4. Run all cells

## Conclusion
Random Forest performed the best with the highest F1-Score of 0.838. The project successfully demonstrates data preprocessing, EDA, model comparison and interpretation.
