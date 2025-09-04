# Self-Healing MLOps System with Drift Detection

## Project Overview
Machine learning models in production degrade over time due to **data drift** and **concept drift**. Most MLOps pipelines deploy and monitor models but do not address automated model recovery and retraining. This project implements a **resilient MLOps system** that detects drift, retrains models automatically, and ensures stable production deployments.

---

## Objectives
- Detect **data drift** and **concept drift** in real-time.
- Build an **automated retraining pipeline** triggered by drift detection.
- Deploy models with **canary testing** and rollback mechanisms.
- Integrate **MLflow Model Registry** for experiment tracking and versioning.
- Develop a **monitoring dashboard** with alerts for system transparency.

---

## Project Phases

### Phase 1: Linear Models (Baseline)
- Train **Logistic Regression** and **Linear Regression** models.
- Preprocess data: missing value imputation, scaling, encoding.
- Evaluate baseline performance with metrics (accuracy, F1, MSE, R²).
- Log models in **MLflow**.

### Phase 2: Non-Linear Models
- Train **Random Forest Classifier/Regressor**.
- Compare performance with baseline linear models.
- Select **best model** based on evaluation metrics.
- Log retrained models to **MLflow registry**.

### Phase 3: Drift Detection
- Implement **Evidently AI** to detect data and target drift.
- Generate drift reports and visualize drifted features.
- Trigger alerts when drift exceeds threshold.

### Phase 4: Self-Healing System
- Automatically retrain models upon drift detection.
- Update preprocessing pipelines.
- Log new models and metrics to MLflow.
- Ensure stable, production-ready deployment.

---

## Dataset
- Example datasets:  
  - Bank Customer Churn Dataset  
  - Credit Card Fraud Detection Dataset  
- Can work with **any structured dataset** that evolves over time.

---

## Key Features
- **Automated drift detection** using Evidently AI.
- **Self-healing pipeline**: auto-retraining + model versioning.
- **Model registry** with MLflow for tracking experiments.
- **Performance monitoring** and alerts for real-time reliability.
- Supports **linear and non-linear models** for classification and regression.
- **Visualization** of feature drift, model comparison, and dataset statistics.

---

## Technologies & Tools
- **Python**, **Pandas**, **NumPy**, **Scikit-learn**
- **MLflow** for model tracking and registry
- **Evidently AI** for drift detection
- **Matplotlib**, **Seaborn**, **Plotly** for visualization
- **Docker** & **Kubernetes** (optional for deployment)
- **Prometheus** & **Grafana** (optional for monitoring)

---

## Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/self-healing-mlops.git
cd self-healing-mlops

# Install required libraries
pip install -r requirements.txt
