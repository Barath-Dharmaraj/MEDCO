from flask import Flask, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

# =========================
# LOAD MODEL + FILES
# =========================
MODEL_PATH = "models/symptom/model.pkl"
SYMPTOMS_PATH = "models/symptom/symptoms_list.pkl"
QUESTIONS_PATH = "models/symptom/symptom_questions.json"

model = pickle.load(open(MODEL_PATH, "rb"))
symptoms_list = pickle.load(open(SYMPTOMS_PATH, "rb"))

# load chatbot questions
with open(QUESTIONS_PATH, "r") as f:
    symptom_questions = json.load(f)


# =========================
# HELPER: Convert symptoms → vector
# =========================
def symptoms_to_vector(user_symptoms):
    # normalize symptom list once
    normalized_list = [s.strip().lower() for s in symptoms_list]

    vector = [0] * len(symptoms_list)

    for s in user_symptoms:
        s_norm = s.strip().lower()
        if s_norm in normalized_list:
            idx = normalized_list.index(s_norm)
            vector[idx] = 1

    return np.array(vector).reshape(1, -1)

# =========================
# ROUTE: Get all symptoms
# =========================
@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": symptoms_list})


# =========================
# ROUTE: Predict disease
# =========================
@app.route("/predict_symptom", methods=["POST"])
def predict_symptom():
    data = request.get_json()

    if not data or "symptoms" not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    user_symptoms = data["symptoms"]

    vec = symptoms_to_vector(user_symptoms)

    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    results = []
    for i in range(len(classes)):
        results.append({
            "disease": classes[i],
            "probability": float(round(probs[i], 3))
        })

    results = sorted(results, key=lambda x: x["probability"], reverse=True)

    return jsonify({"predictions": results[:3]})


# =========================
# ROUTE: Chatbot
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"reply": "Please describe your symptoms."})

    text = data["message"].lower()

    detected = []
    for s in symptoms_list:
        if s in text:
            detected.append(s)

    if not detected:
        return jsonify({
            "reply": "I could not detect symptoms. Please mention symptoms like fever, cough, headache."
        })

    vec = symptoms_to_vector(detected)
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    top_idx = np.argmax(probs)
    disease = classes[top_idx]
    prob = round(float(probs[top_idx]), 2)

    reply = f"Based on your symptoms, you may have {disease} ({prob*100:.1f}%)."

    return jsonify({
        "reply": reply,
        "detected_symptoms": detected
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
