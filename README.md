# Diabetes Onset Prediction Using the Pima Indians Dataset

CLA-3 Machine Learning case study (classification).

Predicts five-year diabetes onset from 8 clinical measurements (768 patients) using Logistic Regression, Random Forest and k-NN.

## Run
```
pip install -r requirements.txt
python diabetes_prediction.py
```
Metrics and figures are written to `results/` (`metrics.csv`, `lr_coefficients.csv`, `examples.csv`, `roc.png`, `lr_coef_confusion.png`, `run_log.txt`).

## Data
`data/diabetes.csv` – Pima Indians Diabetes Database (UCI / NIDDK), 768 rows.

## Key results (192-patient test set, seed 42)
| Model | Accuracy | Recall | Specificity | Test AUC |
|---|---|---|---|---|
| Logistic Regression | 72.4% | 50.7% | 84.0% | 82.4% |
| Random Forest | 74.5% | 55.2% | 84.8% | 81.1% |
| k-NN (k=7) | 72.4% | 52.2% | 83.2% | 78.2% |

Base paper: Smith et al., "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus", 1988.
