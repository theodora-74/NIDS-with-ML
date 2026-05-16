# Adaptive ML-NIDS

An Adaptive Machine Learning Network Intrusion Detection System for Zero-Day Attack Identification through Ensemble Classification and Anomaly Detection.

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 99.83% |
| Precision | 0.9980 |
| Recall | 0.9983 |
| F1-Score | 0.9981 |
| Test Samples | 25,195 |
| Attack Types | 23 |
| Misclassifications | 43 (0.17%) |
| Training Time | 9 min 27 sec |

## Architecture

### Supervised Ensemble (Weighted Voting)
- XGBoost — 30%
- LightGBM — 30%
- Random Forest — 25%
- Gradient Boosting — 15%

### Anomaly Detection (Trained on 53,874 normal samples)
- Isolation Forest — path-length anomaly scoring
- One-Class SVM — normal behaviour boundary
- PyTorch Autoencoder (41→20→10→20→41) — reconstruction error threshold: 0.219

### Hybrid Decision Fusion
- If ensemble confidence < 0.7 AND anomaly score > 0.5 → ZERO-DAY ALERT
- Otherwise → ensemble classification accepted

## Tech Stack
- Ubuntu Server 22.04 LTS (VirtualBox 7.0, 4GB RAM, 2 CPU)
- Python 3.10
- XGBoost 1.7.6, LightGBM 4.0, scikit-learn 1.3
- PyTorch 2.1 (selected over TensorFlow due to AVX incompatibility)
- River 0.18 (ADWIN concept drift detection)
- Scapy 2.5 (real-time packet capture)
- SHAP 0.42 (explainability)

## Dataset
- NSL-KDD: 148,517 total records
- 41 features (38 numeric + 3 categorical)
- 23 individual attack types
- Train: 100,778 / Test: 25,195 (80/20 stratified split)

## Setup

```bash
git clone https://github.com/theodora-74/NIDS-with-ML.git
cd NIDS-with-ML
python3 -m venv nids_env
source nids_env/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python3 main.py train --dataset nslkdd --data-path data/datasets/NSL-KDD
```

## Real-Time Detection

```bash
sudo python3 main.py realtime eth0
```

## Project Structure
adaptive_nids/
├── main.py                          # CLI entry point
├── requirements.txt                 # Dependencies
├── config/config.yaml               # System configuration
├── src/
│   ├── data/dataset_loader.py       # Multi-dataset fusion
│   ├── features/feature_processor.py # StandardScaler + LabelEncoder
│   ├── detection/
│   │   ├── ensemble_classifier.py   # 4-model weighted voting
│   │   ├── anomaly_detector.py      # IF + SVM + Autoencoder
│   │   └── hybrid_detector.py       # Decision fusion logic
│   ├── learning/                    # River + ADWIN drift detection
│   ├── realtime/packet_capture.py   # Scapy live capture
│   └── utils/                       # Config, logging, metrics
└── output/reports/                  # Results and confusion matrix

## Author

Filaj Theodora — BSc Computing, University of Greater Manchester (2026)

Undergraduate Research Project — New York College Thessaloniki
