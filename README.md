# 🚀 DataML Pro — Smart Data Analytics & Machine Learning Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-data-analytics-pro.streamlit.app/)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-green.svg)
![Grafana](https://img.shields.io/badge/Grafana-11.1+-f89406.svg)

> **DataML Pro** is an end-to-end, enterprise-grade Data Analytics and Automated Machine Learning (AutoML) platform built with Streamlit, Scikit-Learn, XGBoost, PostgreSQL, and Grafana. Turn any raw CSV dataset into clean insights, trained ML models, interactive dashboards, and downloadable prediction artifacts in minutes.

---

## 🌟 Key Features & Pipeline Stages

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Upload   │ ──> │   2. Clean   │ ──> │ 3. Preprocess│ ──> │ 4. Train ML  │ ──> │ 5. Dashboard │
│   Dataset    │     │   & EDA      │     │ & Engineers  │     │  & Evaluate  │     │ & Predictions│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 📂 1. Flexible CSV Upload & Inspection
- **Encoding Fallback Engine**: Auto-detects and loads files encoded in `UTF-8`, `Latin-1`, or `Windows-1252` (CP1252).
- **Dataset Summary**: Dynamic view of rows, columns, data types, and memory usage.

### 🧹 2. Automated Data Cleaning & EDA
- **Smart Imputation**: Handles missing numerical and categorical values intelligently.
- **Outlier Detection & Capping**: IQR (Interquartile Range) boundary detection and capping.
- **Exploratory Data Analysis (EDA)**: Correlation heatmaps, distribution plots, box plots, count charts, and missing value heatmaps powered by Plotly.

### ⚙️ 3. Feature Engineering & Preprocessing
- **Categorical Encoders**: One-Hot Encoding and Label Encoding with target mapping exports.
- **Numerical Scalers**: Choice of `StandardScaler` or `RobustScaler`.
- **Target Detection**: Auto-detects classification vs. regression tasks.
- **Split Management**: Automated Train/Test partitioning with fixed random seeds.

### 🤖 4. High-Performance Machine Learning Engine
- **10 Benchmark Models**:
  - **Regression**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
  - **Classification**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
- **Performance Optimizations**:
  - **Gradient Boosting**: Configured with early stopping (`n_iter_no_change=10`), stochastic subsampling (`0.8`), and max depth limits for **~60% faster training**.
  - **Cross-Validation**: 3-fold CV evaluation with real-time UI progress updates.
- **Leaderboard Metrics**: Evaluates models across R², RMSE, MAE, F1-Score, Accuracy, Precision, Recall, and ROC AUC.

### 📊 5. Dual Analytics Dashboards (Streamlit + Grafana)
- **✨ Built-in Live Analytics**: Native interactive Plotly charts rendering Executive Summary, Leaderboard, Top 15 Feature Importances, Correlation Heatmap, and Actual vs. Predicted plots directly in the web app (works 100% online & offline).
- **📈 PostgreSQL & Grafana Bridge**: Synchronizes data to PostgreSQL tables (`core/data_bridge.py`) with an automated 18-panel Grafana dashboard across 5 categories.

### 🎯 6. Prediction Engine & Artifact Export
- **Batch CSV Inference**: Upload new unlabeled CSVs for instant bulk scoring.
- **Single-Sample Scoring**: Interactive form input for single-instance predictions.
- **Artifact Export**: Downloads winning pipelines using Joblib for deployment.

---

## 📁 Repository Structure

```
smart-data-analytics/
├── app.py                      # Main Streamlit web application & page router
├── runtime.txt                 # Streamlit Cloud Python runtime configuration (Python 3.12)
├── setup_grafana.ps1           # Windows PowerShell setup script for Grafana MSI
├── start_portable_grafana.ps1  # Script to download, extract, & launch Portable Grafana
├── update_grafana_config.ps1   # Script to update Grafana provisioning YAML/JSON configs
├── requirements.txt            # Python dependencies
├── core/                       # Core modular business logic
│   ├── cleaner.py              # Missing values, outliers, & data cleaning logic
│   ├── data_bridge.py          # PostgreSQL schema manager & Grafana sync engine
│   ├── feature_engineering.py  # Categorical encoding & feature extraction
│   ├── loader.py               # Robust CSV parsing & fallback encoding
│   ├── model_training.py       # AutoML model comparison, CV, & fast training loop
│   ├── predictor.py            # Prediction pipeline for test set & bulk CSVs
│   ├── preprocessor.py         # Sklearn pipeline builder & scaling logic
│   ├── themes.py               # UI styling, custom CSS, & dark/light theme tokens
│   ├── utils.py                # Shared helpers, problem type detection, & logger
│   └── visualizer.py           # Plotly chart generators
├── grafana/                    # Grafana configuration & dashboard JSON
│   ├── dashboards/             # Pre-built analytics dashboard JSON schemas
│   └── provisioning/           # Auto-provisioning configs for datasources & dashboards
└── artifacts/                  # Saved Joblib pipeline artifacts
```

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- **Python 3.12+**
- **Git**
- Optional: **PostgreSQL** and **Grafana** (for local Grafana integration)

### 1. Clone the Repository
```bash
git clone https://github.com/kgunashekarnaidu/smart-data-analytics.git
cd smart-data-analytics
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Setting Up Local Grafana Integration (Optional)

To enable the live Grafana dashboard locally on port 3000:

1. **Start Portable Grafana** (No Administrator privileges required):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start_portable_grafana.ps1
   ```
2. **Open the Web App**: Navigate to the **📊 Grafana & Live Analytics Dashboard** page.
3. **Sync Data**: Click **🚀 Sync Data to Grafana / PostgreSQL**.
4. **View Dashboards**: Click the **📊 Embedded Grafana** tab to view your 18 live analytics panels on `http://localhost:3000`.

---

## 🌐 Cloud Deployment (Streamlit Cloud)

This app is pre-configured for **Streamlit Cloud**:
- `runtime.txt` pins **Python 3.12** for optimal compatibility.
- Environment variables (`PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`) can be configured via **Streamlit Secrets**.
- If external PostgreSQL/Grafana is not configured, the **✨ Live Analytics (Built-in)** tab automatically displays all interactive Plotly charts natively without needing external services.

---

## 🏆 Model Benchmarks & Metrics

| Problem Type | Model | Evaluation Metrics |
|---|---|---|
| **Regression** | Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost | R², RMSE, MAE |
| **Classification** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost | Accuracy, F1-Score, Precision, Recall, ROC AUC |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 👨‍💻 Author & Acknowledgments

- **Developer**: [Gunashekar Naidu](https://github.com/kgunashekarnaidu)
- **Repository**: [smart-data-analytics](https://github.com/kgunashekarnaidu/smart-data-analytics)
- **Live Demo**: [https://smart-data-analytics-pro.streamlit.app/](https://smart-data-analytics-pro.streamlit.app/)
