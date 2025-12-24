# QoS/QoE Prediction in Mobile Networks using Machine Learning

## Project Overview
This project aims to predict the Quality of Experience (QoE) of mobile users (specifically for video streaming services like YouTube/Netflix) based on network Quality of Service (QoS) metrics.
Using Machine Learning techniques, the system classifies user experience into three categories: Good, Medium, or Bad, without relying solely on subjective user feedback. The project compares two different algorithms: Random Forest and Multi-Layer Perceptron (MLP/Neural Network).

## Authors
1. Mehmet Nuri Başa - 210104004060
2. Umut Hüseyin Satır - 210104004074

## Project Architecture
The project follows a structured machine learning pipeline:
*   **Data Collection:** QoS metrics (RTT, Throughput, etc.) and Video metrics.
*   **Preprocessing:** Data cleaning, One-Hot Encoding, and Scaling.
*   **Labeling:** Deriving QoE labels (Good/Bad/Medium) from resolution, bitrate, and stalling events.
*   **Training:** Training Random Forest and MLP models.
*   **Evaluation:** Testing models using Accuracy, Precision, Recall, and Confusion Matrices.

## Directory Structure
```plaintext
qoe-prediction-model/
├── data/
│   ├── session-dataset.csv    # Dataset containing QoS and Video metrics
│   └── session-dataset.txt    # Dataset metadata
├── models/
│   ├── features.pkl           # List of feature names used for training
│   ├── mlp_model.pkl          # Trained Neural Network model
│   ├── rf_model.pkl           # Trained Random Forest model
│   └── scaler.pkl             # StandardScaler object for data normalization
├── src/
│   ├── evaluate.py            # Script to test models and visualize Confusion Matrix
│   └── train.py               # Script to preprocess data and train models
├── main.py                    # Analysis script for Feature Importance (Random Forest)
└── .gitignore                 # Files ignored by Git
```

## Prerequisites & Installation
The project is built using Python. You need to install the following libraries to run the code:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

## How to Run the Project
Please follow the steps below in order.

### Step 1: Train the Models
Navigate to the `src` folder and run the training script. This will process the data, train both Random Forest and MLP models, and save them to the `models/` directory.
```bash
cd src
python train.py
```
**Output:** Terminal will display classification reports for both models.

### Step 2: Evaluate the Models
Run the evaluation script to test the saved models on the test dataset. This will display the Confusion Matrix for both models side-by-side.
```bash
python evaluate.py
```
**Output:** A window with Confusion Matrix plots and a terminal classification report.

### Step 3: Feature Importance Analysis
Return to the root directory and run `main.py` to see which network parameters affect the user experience the most.
```bash
cd ..
python main.py
```
**Output:** A bar chart showing the top 20 most influential QoS features (e.g., Network Throughput, RTT).

## Methodology Details

### 1. Data Labeling (QoE Calculation)
Since raw QoS data does not always have a direct "Good/Bad" label, we calculated a composite score based on:
*   **Resolution:** Higher resolution = Higher score.
*   **Bitrate:** Higher bitrate = Higher score.
*   **Stalling:** Presence of stalling = Significant score penalty.

**Classes:**
*   **Bad (0):** Score $\le$ 0
*   **Medium (1):** 0 < Score $\le$ 2
*   **Good (2):** Score > 2

### 2. Models Used
*   **Random Forest Classifier:** Selected for its high accuracy on structured data and ability to provide feature importance.
*   **MLP Classifier (Neural Network):** Used as a comparative model to benchmark performance against decision-tree-based approaches.

## Results
*   **Accuracy:** The Random Forest model achieved approximately 95% accuracy on the test set.
*   **Key Findings:** Network throughput and Round Trip Time (RTT) were identified as the most critical factors influencing video streaming quality.

## License
This project was developed for the Mobile Communications course.
