# 🏥 Disease Prediction — CodeAlpha Internship Task 4

## 📌 Objective
Predict diabetes in patients using machine learning classification algorithms.

## 📊 Dataset
- **Name:** Pima Indians Diabetes Dataset
- **Rows:** 768 patients
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (0=Non-Diabetic, 1=Diabetic)

## 🤖 Models Used
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 70.78% | 0.8152 |
| Decision Tree | 76.62% | 0.7799 |
| Random Forest | 75.97% | 0.8218 |
| Gradient Boosting | 75.97% | 0.8283 ← BEST |
| SVM | 73.38% | 0.7939 |

## 🏆 Best Model
- **Gradient Boosting**
- **ROC-AUC: 0.8283**
- **Accuracy: 75.97%**

## ⚙️ How to Run
pip install numpy pandas matplotlib seaborn scikit-learn

python main.py

## 🛠️ Technologies Used
- Python 3
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

## 👤 Author
- **Name:** Syeda Hijab Zahra
- **Internship:** CodeAlpha Machine Learning Intern
- **GitHub:** https://github.com/syedahijabzahra313-droid

## 🏢 Company
CodeAlpha — https://www.codealpha.tech