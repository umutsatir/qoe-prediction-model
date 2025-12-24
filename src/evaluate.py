import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np

# ------------------------------
# 1. LOAD MODELS AND TOOLS
# ------------------------------
print("Loading models...")
# Load Random Forest Model
rf_model = joblib.load("../models/rf_model.pkl")

# Load Neural Network (MLP) Model
try:
    mlp_model = joblib.load("../models/mlp_model.pkl")
except:
    print("WARNING: mlp_model.pkl not found. Only RF will be tested.")
    mlp_model = None

scaler = joblib.load("../models/scaler.pkl")
features = joblib.load("../models/features.pkl")

# ------------------------------
# 2. PREPARE DATA (Test Data)
# ------------------------------
df = pd.read_csv("../data/session-dataset.csv")

# QoE Labeling Function (Same as training)
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

    if score <= 0: return 0      # Bad
    elif score <= 2: return 1    # Medium
    else: return 2               # Good

df["label"] = df.apply(compute_qoe, axis=1)

# Feature Selection and Preprocessing
X = df[features]
y = df["label"]
X = X.fillna(0)
categorical_features = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Align columns (According to Scaler's expectation)
if hasattr(scaler, 'feature_names_in_'):
    expected_features = scaler.feature_names_in_
else:
    expected_features = X.columns.tolist()

X = X.reindex(columns=expected_features, fill_value=0)
X_scaled = scaler.transform(X)

# Convert numeric labels to text (For better visualization)
label_map = {0: "Bad", 1: "Medium", 2: "Good"}
y_str = [label_map[y_val] for y_val in y]

# ------------------------------
# 3. EVALUATION AND VISUALIZATION
# ------------------------------

# Create area to plot graphs side by side (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- A) Random Forest Evaluation ---
print("\n" + "="*40)
print(" EVALUATION: RANDOM FOREST ")
print("="*40)
rf_preds = rf_model.predict(X_scaled)
rf_preds_str = [label_map[p] for p in rf_preds]

print(classification_report(y_str, rf_preds_str))

# Plot Graph (Left side: axes[0])
cm_rf = confusion_matrix(y_str, rf_preds_str, labels=["Bad", "Medium", "Good"])
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Bad", "Medium", "Good"])
disp_rf.plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Random Forest Confusion Matrix')

# --- B) Neural Network (MLP) Evaluation ---
if mlp_model:
    print("\n" + "="*40)
    print(" EVALUATION: NEURAL NETWORK (MLP) ")
    print("="*40)
    mlp_preds = mlp_model.predict(X_scaled)
    mlp_preds_str = [label_map[p] for p in mlp_preds]

    print(classification_report(y_str, mlp_preds_str))

    # Plot Graph (Right side: axes[1])
    cm_mlp = confusion_matrix(y_str, mlp_preds_str, labels=["Bad", "Medium", "Good"])
    disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=["Bad", "Medium", "Good"])
    disp_mlp.plot(ax=axes[1], cmap='Purples') # Use different color
    axes[1].set_title('Neural Network (MLP) Confusion Matrix')
else:
    axes[1].axis('off') # Leave right side empty if MLP is missing

plt.tight_layout()
plt.show()