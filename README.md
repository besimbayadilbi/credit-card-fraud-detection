# Credit Card Fraud Detection

Machine learning system that detects fraudulent credit card transactions with 90% recall.

## 🎯 Results
- **Recall:** 90% (catches 9 out of 10 fraudulent transactions)
- **Precision:** 88% (low false alarm rate)
- **Model:** XGBoost with SMOTE balancing
- **Dataset:** 284,807 transactions with 0.17% fraud rate

## 🛠️ Tech Stack
- Python, Pandas, NumPy, Matplotlib
- scikit-learn, XGBoost, imbalanced-learn
- Jupyter Notebook

## 📊 Key Techniques
- SMOTE for handling severe class imbalance
- Precision-Recall optimization for imbalanced data
- Threshold tuning for business requirements
- Confusion matrix analysis

## 🚀 What I Learned
- Handling imbalanced datasets (fraud is only 0.17% of data)
- Why accuracy is misleading for rare events
- Precision vs Recall tradeoffs
- Feature importance analysis with XGBoost

## 📦 Files
- `fraud_detection.ipynb` - Complete analysis and model training
- `fraud_detector_xgboost.pkl` - Trained XGBoost model (ready for deployment)
- `scaler.pkl` - StandardScaler for feature preprocessing

## 🔄 How to Use the Model
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('fraud_detector_xgboost.pkl')
scaler = joblib.load('scaler.pkl')

# Make prediction on new transaction
new_transaction = pd.DataFrame({...})  # Your transaction data
scaled_data = scaler.transform(new_transaction)
prediction = model.predict(scaled_data)
fraud_probability = model.predict_proba(scaled_data)[:, 1]

print(f"Fraud: {prediction[0]}")
print(f"Probability: {fraud_probability[0]:.2%}")
```

## 📊 Dataset
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle