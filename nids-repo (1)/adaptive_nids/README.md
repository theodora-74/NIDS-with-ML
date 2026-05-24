# Adaptive ML-NIDS

An Adaptive Machine Learning Network Intrusion Detection System for Zero-Day Attack Identification.

## Results
- Accuracy: 99.83%
- Precision: 0.9980
- Recall: 0.9983
- F1-Score: 0.9981
- 25,195 test samples across 23 individual attack types

## Architecture
- Supervised Ensemble: XGBoost (30%) + LightGBM (30%) + Random Forest (25%) + Gradient Boosting (15%)
- Anomaly Detection: Isolation Forest + One-Class SVM + PyTorch Autoencoder
- Hybrid Decision Fusion: confidence-aware arbitration

## Tech Stack
- Ubuntu Server 22.04 LTS (VirtualBox)
- Python 3.10
- XGBoost, LightGBM, scikit-learn, PyTorch 2.1
- River (ADWIN drift detection), Scapy (real-time capture), SHAP (explainability)

## Setup
python3 -m venv nids_env

source nids_env/bin/activate

pip install -r requirements.txt

python3 main.py train --dataset nslkdd --data-path data/datasets/NSL-KDD

## Author

Filaj Theodora  — BSc Computing, University of Greater Manchester (2026)
