# Python-and-AI-Bootcamp-by-Icodeguru-
# 🏥 Heart Disease Prediction - End-to-End ML Pipeline

A comprehensive machine learning project that demonstrates a complete pipeline from data preprocessing to model deployment for heart disease prediction.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

This project implements a complete machine learning workflow to predict the presence of heart disease in patients using various clinical parameters. The pipeline includes data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and comprehensive evaluation.

## 🎯 Objectives

- ✅ Build an end-to-end machine learning pipeline
- ✅ Perform comprehensive exploratory data analysis (EDA)
- ✅ Compare multiple classification algorithms
- ✅ Optimize models using hyperparameter tuning
- ✅ Evaluate model performance with multiple metrics
- ✅ Identify the most important predictive features

## 📊 Dataset

**Dataset Name**: Heart Disease UCI  
**Source**: [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)  
**Samples**: 303 patients  
**Features**: 13 clinical attributes  
**Target**: Presence of heart disease (0 = No, 1 = Yes)

### Features Description:
- `age`: Age in years
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy (0-3)
- `thal`: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
- `target`: Heart disease presence (0 = no, 1 = yes)

## 🛠️ Technical Stack

### Programming Language
- **Python 3.8+**

### Core Libraries
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Model Training**: KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier

### Development Environment
- **Google Colab** (Jupyter Notebook compatible)

## 📈 Methodology

### 1. Data Preprocessing
- Missing value handling
- Duplicate removal
- Feature scaling using StandardScaler
- Train-test split (80-20)

### 2. Exploratory Data Analysis
- Statistical analysis with NumPy and Pandas
- Correlation heatmaps
- Distribution plots
- Interactive visualizations with Plotly

### 3. Model Training
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Random Forest Classifier**

### 4. Hyperparameter Tuning
- RandomizedSearchCV for optimization
- Parameter grids for each algorithm
- Cross-validation (5-fold)

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- ROC curves and AUC scores
- Feature importance analysis

## 📊 Results

### Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| KNN | 0.85 | 0.86 | 0.86 | 0.86 |
| Decision Tree | 0.90 | 0.91 | 0.90 | 0.90 |
| Random Forest | 0.93 | 0.94 | 0.93 | 0.93 |

### Key Findings
- 🏆 **Best Performing Model**: Random Forest Classifier (93% accuracy)
- 🔍 **Most Important Features**: 
  1. `thalach` (Maximum heart rate)
  2. `oldpeak` (ST depression)
  3. `ca` (Number of major vessels)
- ⚡ **Hyperparameter Tuning Impact**: Average improvement of 4-6% across all models

## 🚀 How to Run

### Google Colab
1. Open the provided notebook in Google Colab
2. Run all cells sequentially
3. The dataset will be automatically downloaded

### Local Jupyter Notebook
```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git

# Install requirements
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook ML_Assignment.ipynb
