import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------
# NOTE: Why Only Random Forest?
# Neural Network (MLP) models do not provide "Feature Importance" information
# directly like Random Forest. Therefore, we use the Random Forest model
# to see the most effective criteria.
# ------------------------------

try:
    # Adjusted file paths
    model = joblib.load("models/rf_model.pkl")
    features = joblib.load("models/features.pkl")
    print("Model and features loaded successfully.")

    # Get feature importances
    importances = model.feature_importances_
    
    # Sort the top 20 features
    # argsort sorts from small to large, we take the last (largest) 20
    idx = np.argsort(importances)[-20:]

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Draw bar chart
    plt.barh(np.array(features)[idx], importances[idx], color='skyblue')
    
    plt.title("Top 20 Factors Affecting User Experience (QoE)", fontsize=14)
    plt.xlabel("Importance Level", fontsize=12)
    plt.ylabel("Network Features (QoS Features)", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7) # Grid lines for easier reading
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("ERROR: Model files not found! Please run 'train.py' first.")
    print("Path searched: ../models/rf_model.pkl")