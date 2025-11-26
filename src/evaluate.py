import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

import pandas as pd
df = pd.read_csv("data/session-dataset.csv")

# Target mapping
def mos_to_qoe(x):
    if x <= 2:
        return "Bad"
    elif x == 3:
        return "Medium"
    else:
        return "Good"

df["label"] = df["MOS_p1203"].apply(mos_to_qoe)

# Apply same preprocessing as training
X = df[features]
y = df["label"]

X = X.fillna(0)
# Apply one-hot encoding for categorical features (same as training)
categorical_features = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Align columns with what the scaler expects (from training)
# Get expected feature names from scaler
if hasattr(scaler, 'feature_names_in_'):
    expected_features = scaler.feature_names_in_
else:
    # Fallback: try to infer from the model or use column order
    expected_features = X.columns.tolist()

# Reindex to match expected features, filling missing columns with 0
X = X.reindex(columns=expected_features, fill_value=0)

X_scaled = scaler.transform(X)

# Prediction
preds_numeric = model.predict(X_scaled)

# Map numeric predictions to string labels (same as y)
label_map = {0: "Bad", 1: "Medium", 2: "Good"}
preds = [label_map[p] for p in preds_numeric]

print(classification_report(y, preds))

cm = confusion_matrix(y, preds, labels=["Bad", "Medium", "Good"])
disp = ConfusionMatrixDisplay(cm, display_labels=["Bad", "Medium", "Good"])
disp.plot()
plt.show()
