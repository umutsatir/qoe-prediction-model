import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_PATH = "data/session-dataset.csv"

def mos_to_qoe(x):
    if x <= 2:
        return 0
    elif x == 3:
        return 1
    else:  # 4 or 5
        return 2


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)

    # ---- Target Label ----
    df["label"] = df["MOS_p1203"].apply(mos_to_qoe)


    # ---- Feature Selection ----
    exclude = [
        "MOS_p1203", "startup_delay_class", "avg_resolution_class",
        "stalling_class", "avg_bitrate_class"
    ]
    FEATURES = [c for c in df.columns if c not in exclude + ["label"]]

    X = df[FEATURES]
    y = df["label"]

    # ---- Preprocessing ----
    X = X.fillna(0)
    # Identify and one-hot encode categorical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, FEATURES

def train():
    X, y, scaler, FEATURES = load_and_prepare()

    # ---- Data split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Model ----
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ---- Save ----
    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(FEATURES, "models/features.pkl")

    # ---- Evaluation ----
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    return model, X_test, y_test, FEATURES

if __name__ == "__main__":
    train()
