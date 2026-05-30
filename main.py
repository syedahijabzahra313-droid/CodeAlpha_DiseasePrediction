# ============================================================
#   CodeAlpha Internship — Task 4: Disease Prediction
#   Dataset : Diabetes (built-in sklearn)
#   File    : main.py
#   Folder  : CODEALPHA_DISEASEPREDICTION/
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, f1_score
)

print("=" * 60)
print("   CodeAlpha — Task 4 : Disease Prediction")
print("   Dataset : Pima Indians Diabetes Dataset")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# ── 1. LOAD DATASET ───────────────────────────────────────────
print("\n[1/6] Loading Dataset...")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

try:
    df = pd.read_csv(url, names=cols)
    print("   ✅ Dataset downloaded from internet!")
except:
    # Fallback synthetic data
    np.random.seed(42)
    n = 768
    df = pd.DataFrame({
        'Pregnancies': np.random.randint(0,17,n),
        'Glucose': np.random.randint(70,200,n),
        'BloodPressure': np.random.randint(40,120,n),
        'SkinThickness': np.random.randint(10,60,n),
        'Insulin': np.random.randint(10,300,n),
        'BMI': np.round(np.random.uniform(18,55,n),1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.1,2.5,n),3),
        'Age': np.random.randint(18,80,n),
        'Outcome': np.random.choice([0,1],n,p=[0.65,0.35])
    })
    print("   ✅ Using synthetic dataset!")

print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print(f"   Diabetic (1)     : {(df['Outcome']==1).sum()}")
print(f"   Non-Diabetic (0) : {(df['Outcome']==0).sum()}")

# ── 2. EDA ────────────────────────────────────────────────────
print("\n[2/6] Exploratory Data Analysis...")

print(df.describe().round(2))

# Class distribution
plt.figure(figsize=(5,4))
df['Outcome'].value_counts().plot(
    kind='bar', color=['#2ecc71','#e74c3c'], edgecolor='black')
plt.xticks([0,1],['Non-Diabetic','Diabetic'], rotation=0)
plt.title('Class Distribution', fontsize=13)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("outputs/01_class_distribution.png", dpi=150)
plt.show()
print("   Saved → outputs/01_class_distribution.png")

# Correlation heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150)
plt.show()
print("   Saved → outputs/02_correlation_heatmap.png")

# Feature distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']
for ax, feat in zip(axes.flatten(), features):
    df[df['Outcome']==0][feat].hist(
        ax=ax, alpha=0.6, color='#2ecc71', label='Non-Diabetic', bins=20)
    df[df['Outcome']==1][feat].hist(
        ax=ax, alpha=0.6, color='#e74c3c', label='Diabetic', bins=20)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=7)
plt.suptitle("Feature Distributions by Class", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/03_feature_distributions.png", dpi=150)
plt.show()
print("   Saved → outputs/03_feature_distributions.png")

# ── 3. PREPROCESSING ──────────────────────────────────────────
print("\n[3/6] Preprocessing...")

# Replace 0s with median (invalid values)
for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Train : {X_train.shape[0]} samples")
print(f"   Test  : {X_test.shape[0]} samples")

# ── 4. MODEL TRAINING ─────────────────────────────────────────
print("\n[4/6] Training Models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name in ["Logistic Regression","SVM"] else X_train
    Xte = X_test_sc  if name in ["Logistic Regression","SVM"] else X_test
    print(f"   Training {name}...", end=" ", flush=True)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:,1]
    results[name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC":  roc_auc_score(y_test, y_prob),
        "F1-Score": f1_score(y_test, y_pred),
    }
    print(f"✅  Accuracy={results[name]['Accuracy']:.4f}")

# ── 5. EVALUATION ─────────────────────────────────────────────
print("\n[5/6] Evaluation...")

best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
best = results[best_name]

print(f"\n   Best Model : {best_name}")
print(f"   ROC-AUC    : {best['ROC-AUC']:.4f}\n")
print(classification_report(y_test, best["y_pred"],
      target_names=["Non-Diabetic","Diabetic"]))

# Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Diabetic","Diabetic"],
            yticklabels=["Non-Diabetic","Diabetic"])
plt.title(f"Confusion Matrix — {best_name}", fontsize=13)
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/04_confusion_matrix.png", dpi=150)
plt.show()
print("   Saved → outputs/04_confusion_matrix.png")

# ── 6. CHARTS ─────────────────────────────────────────────────
print("\n[6/6] Plotting Charts...")

# ROC Curves
plt.figure(figsize=(9,6))
colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, lw=2, color=color,
             label=f"{name} (AUC={res['ROC-AUC']:.3f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/05_roc_curves.png", dpi=150)
plt.show()
print("   Saved → outputs/05_roc_curves.png")

# Model Comparison
metrics_df = pd.DataFrame({
    "Model":    list(results.keys()),
    "Accuracy": [r["Accuracy"] for r in results.values()],
    "ROC-AUC":  [r["ROC-AUC"]  for r in results.values()],
    "F1-Score": [r["F1-Score"] for r in results.values()],
}).set_index("Model")

metrics_df.plot(kind="bar", figsize=(11,5), edgecolor="black",
                color=["#3498db","#e74c3c","#2ecc71"])
plt.title("Model Comparison", fontsize=14)
plt.ylabel("Score")
plt.xticks(rotation=15, ha="right")
plt.ylim(0.4, 1.0)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/06_model_comparison.png", dpi=150)
plt.show()
print("   Saved → outputs/06_model_comparison.png")

# Feature Importance
rf = results["Random Forest"]["model"]
feat_imp = pd.Series(rf.feature_importances_,
                     index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(9,5))
feat_imp.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Feature Importance — Random Forest", fontsize=14)
plt.ylabel("Importance")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png", dpi=150)
plt.show()
print("   Saved → outputs/07_feature_importance.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "="*60)
print("   FINAL RESULTS")
print("="*60)
print(f"{'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10} {'F1':>8}")
print("-"*60)
for name, res in results.items():
    marker = " ← BEST" if name == best_name else ""
    print(f"{name:<22} {res['Accuracy']:>10.4f} {res['ROC-AUC']:>10.4f} {res['F1-Score']:>8.4f}{marker}")
print("="*60)
print(f"""
✅ Task 4 Complete!
   Best Model : {best_name}
   ROC-AUC    : {best['ROC-AUC']:.4f}
   Charts saved in 'outputs/' folder
   GitHub repo: CodeAlpha_DiseasePrediction
""")