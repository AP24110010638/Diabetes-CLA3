"""Diabetes Onset Prediction Using the Pima Indians Dataset (CLA-3 case study).

Run:  python diabetes_prediction.py
Outputs metrics + figures into ./results
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

SEED = 42
df = pd.read_csv("data/diabetes.csv")
print("Shape:", df.shape)
print("Class balance:\n", df["Outcome"].value_counts())

# Physiologically impossible zeros -> missing values
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
missing_counts = {c: int((df[c] == 0).sum()) for c in zero_as_missing}
print("Zeros treated as missing:", missing_counts)
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

X, y = df.drop(columns="Outcome"), df["Outcome"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print("Train/test:", len(X_tr), len(X_te))

def pipe(model, scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps)

models = {
    "Logistic Regression": pipe(LogisticRegression(max_iter=1000, random_state=SEED)),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=300, random_state=SEED), scale=False),
    "k-NN (k=7)": pipe(KNeighborsClassifier(n_neighbors=7)),
}
# Baseline: zeros left untouched, no imputation (to show the effect of cleaning)
raw = pd.read_csv("data/diabetes.csv")
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(raw.drop(columns="Outcome"), raw["Outcome"],
                                              test_size=0.25, stratify=raw["Outcome"], random_state=SEED)
lr_raw = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]).fit(Xr_tr, yr_tr)
raw_acc = accuracy_score(yr_te, lr_raw.predict(Xr_te))

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, m in models.items():
    cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")
    m.fit(X_tr, y_tr)
    fitted[name] = m
    p = m.predict(X_te); s = m.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, p).ravel()
    rows.append(dict(Model=name, Accuracy=accuracy_score(y_te, p), Precision=precision_score(y_te, p),
                     Recall_Sensitivity=recall_score(y_te, p), Specificity=tn / (tn + fp),
                     F1=f1_score(y_te, p), ROC_AUC=roc_auc_score(y_te, s),
                     CV_AUC_mean=cv_auc.mean(), CV_AUC_std=cv_auc.std(), TP=tp, TN=tn, FP=fp, FN=fn))
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv("results/metrics.csv", index=False)
pct = res.copy()
for col in ["Accuracy","Precision","Recall_Sensitivity","Specificity","F1","ROC_AUC","CV_AUC_mean","CV_AUC_std"]:
    pct[col] = (pct[col]*100).round(1)
pct.to_csv("results/metrics_percent.csv", index=False)

# Logistic regression coefficients / worked example
lr = fitted["Logistic Regression"]
coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=X.columns).sort_values(ascending=False)
print("\nStandardised LR coefficients:\n", coefs.round(3))
print("Intercept:", round(float(lr.named_steps["clf"].intercept_[0]), 3))
coefs.to_csv("results/lr_coefficients.csv", header=["coef"])

# Correct and incorrect example from the test set
pr = lr.predict_proba(X_te)[:, 1]; pred = (pr >= 0.5).astype(int)
test = X_te.copy(); test["actual"] = y_te.values; test["pred"] = pred; test["prob"] = pr.round(3)
ok = test[(test.actual == 1) & (test.pred == 1)].sort_values("prob", ascending=False).iloc[0]
bad = test[(test.actual == 1) & (test.pred == 0)].sort_values("prob").iloc[0]
print("\nCORRECT example:\n", ok); print("\nINCORRECT example (missed diabetic):\n", bad)
test.loc[[ok.name, bad.name]].to_csv("results/examples.csv")

# Worked-example numbers for the report: one standardised patient -> probability
x_ex = X_te.loc[[ok.name]]
z = float(lr.decision_function(x_ex)[0]); print("\nWorked example z =", round(z, 3), "p =", round(1/(1+np.exp(-z)), 3))

json.dump(dict(shape=list(df.shape), zeros_as_missing=missing_counts, raw_lr_accuracy=round(raw_acc, 3),
               train=len(X_tr), test=len(X_te)), open("results/summary.json", "w"), indent=2)

# Figures
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(5, 4))
for name, m in fitted.items():
    fpr, tpr, _ = roc_curve(y_te, m.predict_proba(X_te)[:, 1])
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, m.predict_proba(X_te)[:,1])*100:.1f}%)")
ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on held-out test set"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("results/roc.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
coefs.sort_values().plot.barh(ax=axs[0], color="#2a6f97"); axs[0].set_title("Logistic regression coefficients\n(standardised features)")
cm = confusion_matrix(y_te, lr.predict(X_te))
axs[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axs[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
axs[1].set_xticks([0, 1], ["No diabetes", "Diabetes"]); axs[1].set_yticks([0, 1], ["No diabetes", "Diabetes"])
axs[1].set_xlabel("Predicted"); axs[1].set_ylabel("Actual"); axs[1].set_title("Confusion matrix (LR)")
fig.tight_layout(); fig.savefig("results/lr_coef_confusion.png", dpi=200); plt.close(fig)
