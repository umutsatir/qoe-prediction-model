import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_PATH = "data/session-dataset.csv"


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


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)

    # --------------------
    # TARGET LABEL: QoE
    # --------------------
    df["label"] = df.apply(compute_qoe, axis=1)


    # --------------------
    # FEATURE SELECTION
    # --------------------
    # remove QoE-related columns
    exclude = [
        "MOS_p1203",
        "startup_delay_class",
        "avg_resolution_class",
        "stalling_class",
        "avg_bitrate_class",
        "label"
    ]

    FEATURES = [c for c in df.columns if c not in exclude]

    X = df[FEATURES]
    y = df["label"]

    # -------------------------
    # PREPROCESSING
    # -------------------------
    X = X.fillna(0)

    # One-hot encode categorical features
    categorical = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, FEATURES


def train():
    X, y, scaler, FEATURES = load_and_prepare()

    # -------------------------
    # TRAIN / TEST SPLIT
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -------------------------
    # SAVE MODEL COMPONENTS
    # -------------------------
    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(FEATURES, "models/features.pkl")

    # -------------------------
    # EVALUATION
    # -------------------------
    preds = model.predict(X_test)
    print("\n=== CLASSIFICATION REPORT ===\n")
    print(classification_report(y_test, preds, target_names=["Bad", "Medium", "Good"]))

    print("\n=== CONFUSION MATRIX ===\n")
    print(confusion_matrix(y_test, preds))

    return model, X_test, y_test, FEATURES


if __name__ == "__main__":
    train()
