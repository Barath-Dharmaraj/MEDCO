MEDCO — Symptom Diagnosis API (Member 1)

Run:
pip install -r requirements.txt
python app.py

Endpoints:

POST /predict
{
  "symptoms": ["fever","cough"]
}

POST /chatbot
{
  "message":"I have fever"
}
