# 🚗 AI Car Price Prediction Web App

An end-to-end Machine Learning project that predicts used car prices using key features and provides an interactive web interface built with Gradio.

---

## 📌 Overview

This project uses real-world car datasets and applies **fuzzy matching** to merge data from different sources. It then performs feature engineering and trains an **XGBoost regression model** to predict car selling prices.

---

## 🚀 Features

- 🔍 Fuzzy matching for dataset merging  
- 🧠 Feature engineering (Brand Goodwill)  
- ⚡ XGBoost regression model  
- 📊 Model performance metrics (MAE, RMSE, R²)  
- 📈 Visualizations (Actual vs Predicted, Error Distribution)  
- 🌐 Interactive web app using Gradio  

---

## 📊 Model Performance

- **MAE:** 2.02 Lakhs  
- **RMSE:** 3.06 Lakhs  
- **R² Score:** 0.64  

> Model uses only 3 features, so performance is moderate.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- FuzzyWuzzy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Gradio  

---


---

## ▶️ Run Locally

```bash
git clone https://github.com/Maria-Chaudhary/CodeAlpha_CarPricePrediction.git
cd CodeAlpha_CarPricePrediction
pip install pandas numpy scikit-learn xgboost gradio matplotlib fuzzywuzzy python-Levenshtein
python app.py
```

Open in browser:
http://127.0.0.1:7860


##💡 Future Improvements
Add more features (Year, Fuel Type, Transmission)
Improve model accuracy (R² > 0.85)
Deploy app online
Enhance UI/UX
