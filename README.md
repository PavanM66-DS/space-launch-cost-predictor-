# 🚀 Global Space Launch Cost Predictor

A complete machine learning web application that predicts space launch costs based on rocket specifications, environmental factors, and mission parameters. Built for data analytics portfolio presentation and deployed live on the cloud.

---

## 🌟 Live Demo & Portfolio Link
* **Live App:** [View Live Streamlit Web App](YOUR_STREAMLIT_CLOUD_URL_HERE)
* **GitHub Repository:** [PavanM66-DS/space-launch-cost-predictor](https://github.com/PavanM66-DS/space-launch-cost-predictor)

---

## 📊 Project Overview
Predicting space launch economics is critical for aerospace cost-benefit analysis. This project analyzes global space launch data to forecast mission costs dynamically. 

* **Model Used:** `Random Forest Regressor` chosen for its capability to handle complex non-linear feature interactions without manual feature engineering.
* **Pipeline Architecture:** Built using `scikit-learn` Pipelines (`ColumnTransformer`, `StandardScaler`, `SimpleImputer`, and `OneHotEncoder`) to automate preprocessing and completely eliminate data leakage.
* **Model Persistence:** Serialized and compressed using `joblib` into `space_launch_cost_model.pkl`.
* **Frontend Interface:** Built with `Streamlit` to provide a real-time, interactive user dashboard.

---

## 📂 Project Structure
```text
space_launch_project/
│
├── train_model.py                # Backend data pipeline and model training script
├── app.py                        # Frontend Streamlit web application
├── space_launch_cost_model.pkl   # Serialized trained machine learning model
├── requirements.txt              # Cloud dependency configuration
└── README.md                     # Project documentation