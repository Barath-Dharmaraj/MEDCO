# MEDCO – Symptom Disease Prediction (Member-1)

This module of MEDCO predicts possible diseases from user symptoms using machine learning.  
It supports symptom input through text, chatbot, and voice.

The system returns the top 3 most likely diseases with probability.

---

## Features

- Symptom → disease prediction
- Voice symptom input
- Chatbot symptom detection
- REST API using Flask
- Machine learning model (RandomForest)

---

## How to Run

1. Install requirements

pip install -r requirements.txt

2. Run backend

cd backend  
python app.py  

3. Open in browser

http://127.0.0.1:5000/voice

---

## API Example
(BETTER USE POSTMAN FOR THIS)
POST /predict_symptom

{
  "symptoms": ["fever", "cough"]
}

---

## Model

The prediction model is trained on `dataset_weighted.csv`  
and stored as `model.pkl`.

---

## Author

Barath D  
MEDCO Project – Member 1
