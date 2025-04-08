# Credit Card Fraud Detection System

## Overview
This project is a **Credit Card Fraud Detection System** that uses **Machine Learning** to classify transactions as **fraudulent** or **legitimate**. The system is implemented using **Streamlit** for the web interface and **Random Forest & XGBoost** models for prediction.

## Features
- **Real-time fraud detection** using trained ML models.
- **User-friendly UI** built with **Streamlit**.
- **Distance calculation** between merchant and cardholder locations.
- **Machine learning models trained on a credit card transaction dataset**.
- **SMOTE** for handling class imbalance.
- **Feature encoding using LabelEncoder & OneHotEncoder**.
- **Model evaluation using accuracy, precision, recall, and F1-score**.
- **Feature importance visualization**.

## Technologies Used
- **Python** (pandas, numpy, scikit-learn, joblib, imbalanced-learn, geopy)
- **Machine Learning Models** (Random Forest, XGBoost)
- **Data Visualization** (Matplotlib, Seaborn)
- **Web Framework** (Streamlit)

## Dataset Used
https://drive.google.com/file/d/1118Jwzj51KpXd0T5jiebn9ykCygwbkhn/view

## Installation
### Clone the repository
git clone https://github.com/ruchitandel/creditcard_fraud_detection.git

cd credit-card-fraud-detection

### Install dependencies
pip install -r requirements.txt

### Run the application
streamlit run app.py

### File Structure

credit_card_fraud_detection/

│

├── .venv/



│

├── Notebook/

│   └── app.ipynb

│

├── data/

│   └── dataset.csv

│

├── docs/

│   └── credit_card_fraud (1).docx

│

├── models/

│   ├── feature_order1.pkl

│   ├── label_encoders1.pkl

│   ├── onehot_encoder1.pkl

│   ├── random_forest_model1.pkl

│   └── xgboost_model1.pkl

│

├── src/

│   └── app.py

│

├── .gitattributes

├── LICENSE

├── README.md

└── requirements.txt

               

## Model Training Process

### Data Preprocessing

- Extract **hour**, **day**, and **month** from the transaction timestamp.
- Drop irrelevant columns (e.g., `name`, `street`, `state`, etc.).
- Encode categorical features using **LabelEncoder** and **OneHotEncoder**.
- Calculate **geodesic distance** between cardholder and merchant locations.
- Handle class imbalance using **SMOTE**.

### Train Machine Learning Models

- **Random Forest** (100 estimators)
- **XGBoost** (max depth = 5, learning rate = 0.1)

### Evaluate Models

- **Accuracy**, **Precision**, **Recall**, **F1-score**
- **Confusion Matrix** & **ROC Curve**
- **Feature Importance Visualization**

## Fraud Detection Workflow

1. **User enters transaction details**.
2. **Features are preprocessed and encoded**.
3. The transaction data is passed to **Random Forest** and **XGBoost** models.
4. The system predicts whether the transaction is **fraudulent** or **legitimate**.
5. The result is displayed on the **Streamlit UI**.

## Screenshots
![image](https://github.com/user-attachments/assets/973bb3a2-cde7-4493-9ddf-59b459765773)
![image](https://github.com/user-attachments/assets/4de8f389-66b4-4b5f-a178-56b03ef16e0f)

![image](https://github.com/user-attachments/assets/67a4893d-d8fe-4c80-ba81-cb30b5d63e02)
![image](https://github.com/user-attachments/assets/1db22caa-5c7f-41df-b664-c0fc3f1f8330)


https://github.com/user-attachments/assets/905560ee-5de3-4ee8-8687-c3695d1ad8fa







