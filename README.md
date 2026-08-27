# Credit Card Fraud Detection

## Problem Statement
Credit card fraud causes major financial losses to banks and customers every year. Fraudulent transactions are extremely rare compared to normal ones, which makes them hard to catch using simple rule-based systems. This project builds a machine learning system that automatically flags a transaction as fraudulent or normal based on its transaction pattern.

## Objective
To build and compare multiple classification models that can accurately identify fraudulent credit card transactions, with a focus on evaluation metrics suited for highly imbalanced data (Precision, Recall, F1-Score) rather than relying on accuracy alone.

**Problem Type:** Binary Classification
**Domain:** Finance / Fraud Detection

## Dataset
- **Source:** [Credit Card Fraud Detection Dataset – Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807
- **Features:** 30 (V1–V28 anonymized PCA components + Time + Amount)
- **Target Variable:** `Class` (0 = Normal, 1 = Fraud)
- **Note:** Dataset is highly imbalanced — only ~0.17% of transactions are fraudulent.

> The dataset file (`creditcard.csv`) is not included in this repository due to its size. Download it from the Kaggle link above and place it in the project's root folder (or `data/` folder) before running the notebook.

## Technologies Used
- **Language:** Python
- **Environment:** Google Colab / Jupyter Notebook
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- **Deep Learning:** TensorFlow / Keras (Artificial Neural Network)

## Project Workflow
1. Load and inspect the dataset
2. Clean the data (check missing values, remove duplicates)
3. Exploratory Data Analysis (class imbalance, amount/time distribution, correlation heatmap)
4. Feature scaling (StandardScaler on Amount and Time)
5. Train/test split (80/20, stratified)
6. Train four classification models
7. Evaluate each model using Accuracy, Precision, Recall, F1-Score
8. Compare models and select the best one
9. Interpret results using feature importance
10. Demonstrate prediction on unseen test samples

## Models Used
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Artificial Neural Network (ANN)

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9991 | 0.8485 | 0.5895 | 0.6957 |
| Random Forest | 0.9995 | 0.9722 | 0.7368 | **0.8383** |
| Gradient Boosting | 0.9993 | 0.8732 | 0.6526 | 0.7470 |
| ANN | 0.9994 | 0.8780 | 0.7579 | 0.8136 |

**Best Model: Random Forest** (highest F1-Score)

## Key Visualizations
See the `images/` folder for exported charts, including:
- Class distribution (Normal vs Fraud)
- Transaction amount distribution
- Correlation heatmap
- Model comparison (F1-Score)
- Confusion matrix (best model)

## Key Insights
- Accuracy is not a reliable metric here due to extreme class imbalance (99.8% normal transactions).
- Random Forest gave the best overall performance, balancing precision and recall better than the other models.
- The most influential features in predicting fraud were **V17, V14, and V12** — consistent with what the correlation heatmap suggested during EDA.

## How to Install and Run
1. Clone this repository or download the notebook
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset from the [Kaggle link above](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project folder
4. Open `IBM_PROJECT.ipynb` in Jupyter Notebook or upload it to Google Colab
5. Run all cells in order

## Future Improvements
- Handle class imbalance more directly using techniques like SMOTE or class-weighting
- Try additional models (e.g., XGBoost) for comparison
- Deploy the best model as a simple web app for real-time fraud checking
- Perform hyperparameter tuning to further improve Random Forest's recall

## Conclusion
Random Forest performed the best with the highest F1-Score of 0.8383. The project demonstrates the complete machine learning lifecycle — from data preprocessing and EDA to model comparison, evaluation, and interpretation — applied to a real-world, highly imbalanced classification problem.
