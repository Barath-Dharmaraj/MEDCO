from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# =========================
# LOAD MODEL + FILES
# =========================
MODEL_PATH = "models/symptom/model.pkl"
MLB_PATH = "models/symptom/mlb.pkl"
SYMPTOMS_PATH = "models/symptom/symptoms_list.pkl"
DATASET_PATH = "models/symptom/dataset_weighted.csv"

model = pickle.load(open(MODEL_PATH, "rb"))
mlb = pickle.load(open(MLB_PATH, "rb"))
symptoms_list = pickle.load(open(SYMPTOMS_PATH, "rb"))

dataset = pd.read_csv(DATASET_PATH)

# disease → symptoms map
disease_map = dataset.groupby("Disease")["Symptom"].apply(list).to_dict()

# =========================
# HELPER FUNCTIONS
# =========================
def symptoms_to_vector(user_symptoms):
    return mlb.transform([user_symptoms])

def jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a & b) / len(a | b) if (a | b) else 0

def predict_from_symptoms(user_symptoms):
    vec = symptoms_to_vector(user_symptoms)
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    results = []

    for i, disease in enumerate(classes):
        ml_prob = probs[i]
        sim = jaccard(user_symptoms, disease_map.get(disease, []))

        final = (0.6 * ml_prob) + (0.4 * sim)

        results.append({
            "disease": disease,
            "probability": round(float(final), 3)
        })

    results = sorted(results, key=lambda x: x["probability"], reverse=True)
    return results[:3]

# =========================
# ROUTES
# =========================

@app.route("/voice")
def voice():
    return render_template("voice.html")

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": symptoms_list})

@app.route("/predict_symptom", methods=["POST"])
def predict_symptom():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    user_symptoms = [s.strip().lower() for s in data["symptoms"]]
    preds = predict_from_symptoms(user_symptoms)

    return jsonify({"predictions": preds})

@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "No speech text provided"}), 400

    text = data["symptoms"].lower()
    words = text.replace("_", " ").split()

    detected = []
    for s in symptoms_list:
        s_words = s.replace("_", " ").split()
        if any(w in words for w in s_words):
            detected.append(s)

    if not detected:
        return jsonify({"predictions": [], "message": "No symptoms detected"})

    preds = predict_from_symptoms(detected)
    return jsonify({"predictions": preds})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"reply": "Please describe symptoms."})

    text = data["message"].lower()

    detected = [s for s in symptoms_list if s in text]

    if not detected:
        return jsonify({"reply": "Please mention symptoms like fever, cough, headache."})

    preds = predict_from_symptoms(detected)
    top = preds[0]

    reply = f"Based on symptoms, possible {top['disease']} ({top['probability']*100:.1f}%)."

    return jsonify({
        "reply": reply,
        "detected_symptoms": detected,
        "predictions": preds
    })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)