# 🧠 Churn Prediction — Intro to Neural Networks  
### Fully Connected Neural Networks • Backpropagation • Dropout • Scikit-Learn • Keras • TensorFlow • NumPy

This project is an introduction to artificial neural networks using a real-world **bank customer churn prediction** dataset.  
The goal is to explore the complete ML → DL pipeline: baseline models, classical ML methods, deep learning models, and finally a **fully connected neural network implemented manually with NumPy**.

---

## 📌 Project Overview

Customer churn prediction is a common classification task in banking.  
Your objective is to determine whether a customer will leave the bank in the next 3 months.

You will build multiple models:

- Baseline classifier  
- Random Forest  
- Scikit-Learn MLP  
- Keras neural network  
- TensorFlow neural network  
- Custom NumPy neural network (forward pass + backpropagation + dropout)

The dataset includes hundreds of numeric features:  
demographics, financial data, credit info, activity trends, account metrics, and product usage.

---

## 🎯 Project Goals

1. Learn how **Fully Connected Neural Networks (FCNNs)** work.  
2. Train neural networks using **Scikit-learn, Keras, TensorFlow**, and **NumPy**.  
3. Implement FCNN from scratch using NumPy and OOP principles.  
4. Perform data preprocessing, cleaning, and feature engineering.  
5. Tune hyperparameters via Grid Search.  
6. Achieve **AUC ≥ 0.8183** on the test dataset.  
7. Optionally achieve **AUC ≥ 0.83**.

---

## 📂 Project Structure

```
├── data/
│ ├── train.csv
│ ├── test.csv
├── notebooks/
│ ├── 01_baseline_models.ipynb
│ ├── 02_random_forest.ipynb
│ ├── 03_sklearn_mlp.ipynb
│ ├── 04_keras_model.ipynb
│ ├── 05_tensorflow_model.ipynb
│ ├── 06_numpy_neural_network.ipynb
│ └── 07_results_and_comparison.ipynb
├── predictions.csv
└── README.md
```

---

## 🧪 Methods & Models

### ✔ Baselines
- Majority class classifier  
- Random Forest with Grid Search  

### ✔ Neural Network Implementations
- **Scikit-learn:** `MLPClassifier`  
- **Keras:** Sequential FCNN with dropout  
- **TensorFlow:** Custom-defined FCNN  
- **NumPy:** Manual FCNN implementation (matrix math, activations, backprop, dropout)

---

## 🛠 Preprocessing

- Missing value handling  
- Outlier investigation  
- Feature scaling (StandardScaler / MinMaxScaler)  
- Train/validation split with **stratification (80/20)**  
- Feature selection / pruning  
- Hyperparameter search (Grid Search / Random Search)

---

## 📈 Evaluation Metrics

Models are compared using:

- **Accuracy**
- **AUC-ROC** (primary metric)
- Confusion matrix (optional)
- ROC curve (optional)

Final comparison table example:

| Library | Model | Hyperparameters | Accuracy | AUC |
|---------|--------|------------------|----------|------|
| Baseline | Majority | — | X.XX | X.XX |
| Random Forest | Grid Search | depth, trees | X.XX | X.XX |
| Scikit-learn | MLPClassifier | hidden layers, lr | X.XX | X.XX |
| Keras | FCNN | layers, dropout | X.XX | X.XX |
| TensorFlow | FCNN | activations, optimizer | X.XX | X.XX |
| NumPy | Custom FCNN | architecture | X.XX | X.XX |

---

- Models achieving at least **AUC ≥ 0.8183**

---

## 🚀 Achievements

- Implemented FCNNs with 4 different libraries/tools  
- Built a NumPy neural network from scratch using OOP  
- Added dropout and regularization  
- Achieved required AUC on the test set  