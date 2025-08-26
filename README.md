# Pima Indians Diabetes Prediction

## 📌 Project Description
This project uses the **Pima Indians Diabetes Dataset** from Kaggle to predict whether a patient has diabetes based on diagnostic measurements.

## 📊 Dataset
**Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Rows:** 768  
**Columns:** 9 (8 features + 1 target)

**Features:**
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target: 1 = Diabetes, 0 = No Diabetes)

## 🎯 Objective
To build machine learning models that can accurately classify patients as diabetic or non-diabetic.

## 🛠 Steps
1. **Load Data:** Read CSV from Kaggle.
2. **Data Cleaning:** Handle missing values, remove duplicates.
3. **EDA:** Visualize distributions & correlations.
4. **Feature Engineering:** Scaling & preprocessing.
5. **Modeling:** Train KNN, Decision Tree, Random Forest.
6. **Hyperparameter Tuning:** Use RandomizedSearchCV.
7. **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC curve.
8. **Feature Importance:** Identify top contributing features.

## ▶ How to Run
1. Open in **Google Colab**.
2. Upload `kaggle.json` to access dataset.
3. Run the provided Python code cells in order.

## 📈 Results (Example)
| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| KNN                    | 0.77     | 0.75      | 0.73   | 0.74     |
| Decision Tree          | 0.73     | 0.72      | 0.70   | 0.71     |
| Random Forest (Tuned)  | **0.83** | **0.82**  | **0.81** | **0.81** |

## 🏆 Conclusion
Random Forest with tuned hyperparameters performed best, achieving **83% accuracy** and strong F1 score.
