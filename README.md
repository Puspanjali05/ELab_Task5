# ELab_Task5

# 🌳 Decision Tree & Random Forest - Classification

This project is a part of the AI & ML Internship program.  
The objective is to implement **Decision Tree** and **Random Forest** classifiers, compare their performances, control overfitting, and interpret results using feature importance.

---

## 📌 Objective

Train and evaluate classification models to:
- Predict target classes based on input features.
- Visualize and interpret decision-making processes in Decision Trees.
- Compare performance of a single Decision Tree vs an ensemble Random Forest.
- Identify and rank important features contributing to predictions.

---

## 🛠️ Tools & Libraries Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Graphviz / pydotplus (for tree visualization)  
- Joblib (for saving models)

---

## 📂 Dataset

The dataset used is the **[Specify dataset name, e.g., Heart Disease UCI Dataset]**.  
**Target Variable:**  
- `0` → Class 0 (e.g., No disease)  
- `1` → Class 1 (e.g., Disease present)  

**Features:**  
Multiple numerical and/or categorical features depending on dataset.

---

## 📊 Project Process

### 1. Data Loading & Preprocessing
- Loaded dataset using pandas.
- Checked shape, column names, and missing values.
- Encoded categorical features using One-Hot Encoding.
- Train–test split (80% train, 20% test) with stratification.

### 2. Baseline Decision Tree
- Trained `DecisionTreeClassifier` with default parameters.
- Evaluated using accuracy, precision, recall, F1-score, confusion matrix.

### 3. Overfitting Control & Hyperparameter Tuning
- Used `max_depth`, `min_samples_split`, `min_samples_leaf`, and `ccp_alpha` to prune.
- Applied `GridSearchCV` for optimal parameters.

### 4. Random Forest Model
- Trained `RandomForestClassifier` with multiple trees (n_estimators=200).
- Used OOB (Out-of-Bag) score for internal validation.
- Compared metrics with Decision Tree.

### 5. Visualization
- Visualized Decision Tree structure using `plot_tree` and Graphviz.
- Plotted feature importance rankings.
- Generated ROC Curve and calculated AUC score.

---

## 📈 Key Insights
- Random Forest achieved better generalization compared to a single Decision Tree.
- Overfitting in Decision Tree reduced after pruning and limiting depth.
- Feature importance plots provided interpretability for both models.
- ROC-AUC values indicated strong discriminative power.

---

## 📁 Files in Repository

| File Name                      | Description                                         |
|--------------------------------|-----------------------------------------------------|
| `ElevateLabs_Task5.ipynb`      | Full code and visualizations for the project        |
| `heart.csv`                    | Dataset                                             |
| `README.md`                    | Project documentation (this file)                   |
