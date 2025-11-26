import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load("models/rf_model.pkl")
features = joblib.load("models/features.pkl")

importances = model.feature_importances_
idx = np.argsort(importances)[-20:]  # en önemli 20 feature

plt.figure(figsize=(10,6))
plt.barh(np.array(features)[idx], importances[idx])
plt.title("Top 20 Most Important QoS Features")
plt.tight_layout()
plt.show()
