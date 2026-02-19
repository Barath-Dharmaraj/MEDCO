import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# =============================
# 1. LOAD RAW DATASET
# =============================
df = pd.read_csv("dataset.csv")

# First column = disease
diseases = df.iloc[:, 0]

# All symptom columns (text)
symptom_cols = df.columns[1:]

# =============================
# 2. CREATE UNIQUE SYMPTOM LIST
# =============================
all_symptoms = set()

for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())

all_symptoms = sorted(all_symptoms)

# =============================
# 3. CREATE BINARY MATRIX
# =============================
X = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for i in range(len(df)):
    for col in symptom_cols:
        symptom = df.iloc[i][col]
        if pd.notna(symptom):
            X.at[i, symptom] = 1

y = diseases

# =============================
# 4. SAVE SYMPTOM ORDER
# =============================
with open("symptoms_list.pkl", "wb") as f:
    pickle.dump(all_symptoms, f)

print("Symptoms order saved")

# =============================
# 5. TRAIN MODEL
# =============================
model = RandomForestClassifier()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained & saved")
