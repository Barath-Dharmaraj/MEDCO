import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

# =============================
# LOAD DATASET
# =============================
df = pd.read_csv("dataset_weighted.csv")

# group symptoms per disease
grouped = df.groupby("Disease")["Symptom"].apply(list)

diseases = list(grouped.index)
symptom_lists = list(grouped.values)

# =============================
# MULTI-LABEL BINARIZER
# =============================
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptom_lists)
y = diseases

# =============================
# TRAIN MODEL
# =============================
model = LogisticRegression(max_iter=3000)
model.fit(X, y)

# =============================
# SAVE FILES
# =============================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(mlb, open("mlb.pkl", "wb"))
pickle.dump(list(mlb.classes_), open("symptoms_list.pkl", "wb"))

print("✅ MEDCO medical model trained")