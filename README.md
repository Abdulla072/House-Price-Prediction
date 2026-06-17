# 🏠 House Price Prediction API

A complete end-to-end Machine Learning project that predicts house prices using property features such as area, quality, neighborhood, garage capacity, and more.

The project includes:

* Data Ingestion Pipeline
* Data Transformation & Preprocessing
* Model Training
* Prediction Pipeline
* FastAPI REST API
* Interactive Swagger UI Documentation

---

## 📊 Dataset

Dataset: **House Prices - Advanced Regression Techniques**

Source: Kaggle

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib
* Git & GitHub

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── data/
├── src/
│   └── house_price/
│       ├── components/
│       ├── exception/
│       ├── logger/
│       ├── pipeline/
│       └── utils/
│
├── app.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Abdulla072/House-Price-Prediction.git
```

Move into the project directory:

```bash
cd House-Price-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Git Bash

```bash
source venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package:

```bash
pip install -e .
```

---

## 🚀 Train the Model

Run the training pipeline:

```bash
python -m house_price.pipeline.training_pipeline
```

The trained model and preprocessor will be saved in the `artifacts/` directory.

---

## 🌐 Run the API

Start the FastAPI server:

```bash
python -m uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 📌 API Endpoints

### GET /

Checks whether the API is running.

Response:

```json
{
  "message": "House Price Prediction API is running"
}
```

### POST /predict

Predicts the house price from user input features.

Example response:

```json
{
  "predicted_price": 208450.75
}
```

---

## 📈 Model Performance

* Mean Absolute Error (MAE): **17,328.72**
* R² Score: **0.8985**

---

## 👨‍💻 Author

**Abdulla Al Biswas**

GitHub: https://github.com/Abdulla072
