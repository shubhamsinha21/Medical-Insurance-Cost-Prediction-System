# 💰 Medical Insurance Cost Prediction using Machine Learning (XGBoost)

An end-to-end Machine Learning project that predicts **medical insurance charges** based on user attributes like age, BMI, smoking status, and more.  
The project includes **data preprocessing, feature engineering, model training, evaluation, and a deployed Streamlit web app**.

---

## 🚀 Live Demo Image
👉 <img width="1510" height="866" alt="ui" src="https://github.com/user-attachments/assets/f485bdc5-d45a-4bb3-8301-71b503e708a9" />

---

## 📊 Problem Statement

Medical insurance companies need to estimate the cost of insurance premiums based on customer health and lifestyle factors.  
This project builds a predictive system to estimate insurance charges using machine learning.

---

## 📂 Dataset

- Source: Kaggle Insurance Dataset
- Features include:
  - Age
  - Sex
  - BMI
  - Number of children
  - Smoking status
  - Region
  - Insurance charges (target variable)

---

## 🧠 ML Pipeline Overview

### 1. Data Preprocessing
- Handled categorical variables:
  - Label Encoding (sex, smoker)
  - One-hot encoding (region, BMI category)
- Feature scaling using StandardScaler
- Removed unnecessary features after correlation analysis

---

### 2. Feature Selection
- Pearson Correlation for numerical features
- Chi-square test for categorical features
- Selected most relevant features affecting insurance charges

---

### 3. Models Trained

Multiple regression models were tested:

- Linear Regression (baseline)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR
- ⭐ **XGBoost Regressor (Best Model)**

---

## 🏆 Final Model Performance

| Model | R² Score |
|------|--------|
| XGBoost | **~0.90** |
| Gradient Boosting | ~0.89 |
| Random Forest | ~0.87 |

### Final Selected Model:
👉 **XGBoost Regressor**

Reason:
- Highest accuracy
- Best generalization
- Stable performance on unseen data

---

## 📈 Evaluation Metrics

- R² Score: ~0.90  
- MAE: ~2500–3000  
- RMSE: ~4200–4500  

These metrics indicate strong predictive performance for real-world insurance estimation.

---

## ⚙️ Hyperparameter Tuning

Performed using **RandomizedSearchCV**:

Parameters tuned:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree

Result:
- Improved model stability
- Slight refinement in RMSE and MAE

---

## 🧪 Tech Stack

- Python 🐍
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit (Deployment UI)
- Joblib (Model saving)

---

## 🖥️ Web App Features

- User input form (age, BMI, smoking, etc.)
- Real-time insurance cost prediction
- Feature importance visualization
- PDF report download
- Clean interactive UI

---

## 📦 Project Structure
insurance-project/
│
├── app.py
├── insurance_xgb_model.pkl
├── model_features.pkl
├── requirements.txt
├── insurance.csv
├── insurancechargeprediction.ipynb
├── .gitignore

---

## 📊 Feature Importance Insight

Most important factors affecting insurance cost:

- 🚬 Smoking status (highest impact)
- 📊 BMI
- 🎂 Age
- 👶 Number of children
- 🌍 Region (minor impact)

---

## 📄 Output Example

```

Input:
Age: 35
BMI: 28
Smoker: Yes

Predicted Insurance Cost:
₹ 25,000 – ₹ 35,000 approx

## 🚀 How to Run Locally

git clone https://github.com/your-username/repo-name.git
cd repo-name
pip install -r requirements.txt
streamlit run app.py

