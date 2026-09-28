# 🏥 Insurance Premium Prediction

An end-to-end machine learning project that predicts insurance premiums from personal and health details. It includes a trained model, a **FastAPI** backend, and an interactive **Streamlit** frontend.



![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)




![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)




![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)




![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)



---

## 📌 Overview

Insurance companies price premiums based on many factors. This project trains a machine learning model on historical data to estimate a premium instantly, and serves it through a REST API and a simple web app.

## ✨ Features

- 📊 Data analysis and model training in a Jupyter Notebook
- 🤖 Trained model saved as `Model.pkl`
- ⚡ FastAPI backend serving real-time predictions
- 🎨 Streamlit frontend for easy, user-friendly input
- 🔌 Frontend and backend connected through API calls

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `Insurance.ipynb` | Data exploration, preprocessing, model training and evaluation |
| `insurance_premium_dataset.csv` | Dataset used to train the model |
| `Model.pkl` | Trained model saved with pickle |
| `Fastapi.py` | FastAPI backend that loads the model and serves predictions |
| `fronted.py` | Streamlit frontend where users enter details and get a prediction |

## 🧠 How It Works

1. **Data**: the CSV is loaded and cleaned in the notebook.
2. **Training**: features are preprocessed and a model is trained and evaluated.
3. **Saving**: the best model is exported to `Model.pkl`.
4. **API**: `Fastapi.py` loads the model and exposes a prediction endpoint.
5. **UI**: `fronted.py` collects user input, sends it to the API, and shows the predicted premium.

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** Pandas, NumPy, Scikit-learn
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Aman-kumarsah/<your-repo-name>.git
cd <your-repo-name>
