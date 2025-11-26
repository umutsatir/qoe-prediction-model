import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

import pandas as pd
df = pd.read_csv("data/session-dataset.csv")

# ------------------------------
#  QoE LABEL FROM COMPOSITE SCORE
# ------------------------------
def compute_qoe(row):
    class_mapping = {'low': 0, 'medium': 1, 'high': 2}
    resolution_class = class_mapping.get(row["avg_resolution_class"], row["avg_resolution_class"])
    bitrate_class = class_mapping.get(row["avg_bitrate_class"], row["avg_bitrate_class"])
    stalling_class = class_mapping.get(row["stalling_class"], row["stalling_class"])
    score = (
        int(resolution_class)
        + int(bitrate_class)
        - int(stalling_class)
    )

    if score <= 0:
        return 0        # Bad
    elif score <= 2:
        return 1        # Medium
    else:
        return 2        # Good

df["label"] = df.apply(compute_qoe, axis=1)

print(df["label"].value_counts())
# Apply same preprocessing as training
X = df[features]
y = df["label"]  # y is numeric (0, 1, 2)

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
preds = model.predict(X_scaled)

# Map numeric labels to strings for display
label_map = {0: "Bad", 1: "Medium", 2: "Good"}
y_str = [label_map[y_val] for y_val in y]
preds_str = [label_map[p] for p in preds]

print(classification_report(y_str, preds_str))

cm = confusion_matrix(y_str, preds_str, labels=["Bad", "Medium", "Good"])
disp = ConfusionMatrixDisplay(cm, display_labels=["Bad", "Medium", "Good"])
disp.plot()
plt.show()
