Project Title: Network Anomaly Detection using Deep Learning and "DNA Fingerprinting"

## Recent Enhancements (2025)

### 1. Time-Series Dimension
Network attacks are often sequential. The project now supports a Recurrent Autoencoder (LSTM-based) for time-series anomaly detection. Instead of treating flows independently, you can input sequences of consecutive flow vectors to capture temporal context.

### 2. Threshold Tuning
Instead of using a fixed percentile, the anomaly threshold is now tuned using a calibration set of known normal traffic. The threshold is selected to maximize F1-score or achieve a target False Positive Rate (FPR), improving detection reliability.

### 3. Feature Robustness
You can now choose between StandardScaler, RobustScaler, or MinMaxScaler for feature preprocessing. This makes the model more robust to outliers, which are common in network data.

### 4. Baseline Comparison
To demonstrate novelty, the project now includes baseline anomaly detection methods: One-Class SVM and Isolation Forest. Their performance is reported alongside the autoencoder for fair comparison.

---
1. Core Concept: "Network DNA Fingerprinting"
Our project proposes a novel approach to network anomaly detection by treating network traffic flows as biological sequences or "DNA." The goal is to identify malicious activity in encrypted traffic without requiring decryption. We achieve this by analyzing the unique patterns in traffic metadata (features like packet size, timing, and protocol types), which are a fingerprint of the network activity itself. This approach is powerful because it's unsupervised, meaning it doesn't need labeled examples of attacks to learn what constitutes "normal" behavior.

2. Methodology and Implementation
Our methodology is broken down into a four-phase pipeline, each implemented in a separate, modular script for clarity and maintainability.

Phase 1: Feature Encoding (The "DNA" Generation)
This phase is the core of the "fingerprinting" concept. We process raw network flows from a dataset (e.g., UNSW-NB15) and convert them into a machine-readable format.

Process: The data_processor.py script takes raw features from the dataset. It uses a selectable scaler (StandardScaler, RobustScaler, or MinMaxScaler) to normalize numerical features (e.g., duration, packet size) and LabelEncoder to convert categorical features (e.g., protocol types) into numerical tokens. Set the environment variable `SCALER_TYPE` to `robust` or `minmax` to use those scalers.

Output: Each network flow is transformed into a vector of numbers. We refer to this vector as the "pseudo-DNA sequence". This numerical sequence is the input for our deep learning model.

Phase 2: Unsupervised Model Training
In this phase, our AI model learns to recognize the "language" of normal network traffic.

Model: We use a Sparse Autoencoder, a type of neural network with an encoder-decoder architecture. Optionally, a Recurrent Autoencoder (LSTM-based) can be used for time-series anomaly detection.

Encoder: Compresses the "pseudo-DNA" sequences into a low-dimensional representation, forcing the model to capture only the most essential patterns. The activity_regularizer=l1() function on the hidden layers enforces sparsity, which acts as an embedded feature selection mechanism, forcing the model to rely only on the most important features.

Decoder: Attempts to reconstruct the original "pseudo-DNA" sequence from the compressed representation.

Process: The anomaly_detector.py script defines the autoencoder. The main.py script trains this model exclusively on a subset of the dataset known to contain only normal traffic. The goal is to minimize the reconstruction error, which is the difference between the input and the reconstructed output. For time-series, use the LSTM autoencoder.

Phase 3: Anomaly Detection
Once the model is trained, we can use it to identify anomalies in new, unseen traffic.

Process: The main.py script passes new, unanalyzed data to the trained autoencoder. The model attempts to reconstruct these new "pseudo-DNA" sequences.

Output: The model's output is an anomaly score, which is the Mean Squared Error (MSE) of the reconstruction. For normal traffic, this error will be very low because the model has learned its patterns. For anomalous or malicious traffic, the model will struggle to reconstruct the sequence, resulting in a high anomaly score. The threshold is now tuned using a calibration set to maximize F1-score or achieve a target FPR, rather than using a fixed percentile.

Phase 4: Real-Time Anomaly Detection
This phase demonstrates the project's practical utility by extending the framework to live network traffic.

Process: The live_detector.py script is a standalone program that combines a packet capture tool (scapy) with the pre-trained model.

Workflow:

The script continuously sniffs live network packets.

For each packet, it extracts a set of features that match the training data.

It loads the saved StandardScaler object (from models/scaler.joblib) to preprocess the live features correctly.

It feeds the preprocessed data into the saved autoencoder model (models/autoencoder_model.keras).

It calculates the reconstruction error and immediately flags a packet as an anomaly if its score exceeds the pre-defined threshold.

3. Performance and Evaluation
The project's effectiveness is a crucial part of the evaluation.

Metrics: The main.py script calculates and reports standard classification metrics:

Precision: What percentage of flagged anomalies were actually malicious?

Recall: What percentage of actual anomalies did our model successfully find?

F1-Score: The harmonic mean of Precision and Recall, providing a balanced measure.

ROC-AUC: The area under the Receiver Operating Characteristic curve, a robust metric that measures the model's ability to distinguish between normal and anomalous traffic.

Visualization: The main.py script generates and saves a ROC Curve plot to visually demonstrate the model's performance and provide a strong evidence base for the research paper.

Baseline Comparison: The project also reports the performance of One-Class SVM and Isolation Forest as baseline methods for anomaly detection, allowing for a fair comparison and demonstrating the effectiveness of the deep learning approach.

By following this approach, our project provides a robust, scalable, and effective solution for detecting anomalies in encrypted network traffic, which is a significant challenge in modern cybersecurity.

## Project flow & how to run

This section describes the minimal steps to run the project locally, where to put the dataset, and what outputs the pipeline generates.

1. Place your dataset

- Put the pre-split CSV files under `Dataset/raw/Training and Testing Sets/`.
	- Training: `UNSW_NB15_training-set.csv`
	- Testing: `UNSW_NB15_testing-set.csv`

2. Create and activate a Python virtual environment (Windows cmd)

```cmd
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

If the repo contains `requirements.txt` (recommended):

```cmd
pip install -r requirements.txt
```

If not, install the main packages used by the project:

```cmd
pip install tensorflow scikit-learn pandas matplotlib xgboost mlflow
```

4. (Optional) Choose a scaler

Set the `SCALER_TYPE` environment variable to change preprocessing behavior:

```cmd
set SCALER_TYPE=robust   # or minmax
```

5. Run the pipeline

```cmd
python src\main.py
```

6. Outputs (generated)

- `model/autoencoder_model.keras` — trained Keras model
- `model/scaler.pkl` — saved scaler object
- `Dataset/processed/processed_data_train.csv` and `_test.csv` — processed versions of train/test
- `Dataset/results/anomaly_scores.csv` — anomaly scores + predicted labels
- `report/figures/performance_metrics.png` — ROC curve

7. Git / dataset note

The dataset is large and should not be checked into source control. `.gitignore` already includes the `Dataset/` directory and `model/` artifacts to prevent accidental commits. If you have already committed large files, run:

```cmd
git rm -r --cached Dataset
git rm -r --cached model
git commit -m "Remove large dataset and model artifacts; add to .gitignore"
git push
```

8. Troubleshooting

- If Python is not found, use the Python launcher `py -3 -m venv venv` or install Python from https://python.org and check "Add to PATH" during installation.
- If `pip install` fails for some packages, install them one-by-one or consult the package docs for platform-specific builds (e.g., `xgboost`/`tensorflow`).

If you want, I can also add a small `scripts/` helper (e.g., `scripts/run.sh` / `scripts/run.bat`) to automate these commands.