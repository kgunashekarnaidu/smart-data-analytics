# Data Analytics & ML Application

Professional Streamlit dashboard for end-to-end data analytics and machine learning on any CSV dataset.

## Features

- CSV upload and dataset preview
- Automatic data cleaning and outlier handling
- Exploratory data analysis and interactive visualizations
- Feature engineering, encoding, and scaling
- Classification and regression model training with comparison
- Predictions and downloadable results
- Joblib pipeline persistence
- User-selectable **Dark** and **Light** themes (sidebar → Appearance)

## Project Structure

```
data-analytics-ml-app/
├── app.py                  # Streamlit entry point (Step 2)
├── pages/                  # UI pages
├── core/                   # Business logic modules
├── models/                 # Model definitions
├── artifacts/              # Saved pipelines and models
├── logs/                   # Application logs
├── requirements.txt
└── README.md
```

## Setup

```powershell
cd data-analytics-ml-app
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Development Status

| Step | Component              | Status     |
|------|------------------------|------------|
| 1    | Folder structure       | Complete   |
| 2    | app.py                 | Complete   |
| 3    | core/loader.py         | Complete   |
| 4    | core/utils.py          | Complete   |
| 5    | core/cleaner.py        | Complete   |
| 6    | core/feature_engineering.py | Complete   |
| 7    | core/preprocessor.py        | Complete   |
| 8    | core/visualizer.py          | Complete   |
| 9    | core/model_training.py      | Complete   |
| 10   | core/predictor.py           | Complete   |
| —    | pages split & polish          | Optional   |

Built from the `TASK_1.ipynb` retail inventory ML notebook, generalized for any CSV dataset.
