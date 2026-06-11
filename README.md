# Breast Cancer Classification & Model Selection Framework

An end-to-end machine learning pipeline built to train, evaluate, and benchmark multiple classification algorithms to determine the optimal model for predicting whether a breast tumor is benign or malignant. This project implements comprehensive data preprocessing and utilizes a structured grid of performance metrics to ensure maximum diagnostic reliability.

## 📋 Project Overview

In medical diagnostics, predictive accuracy and minimizing false negatives (missing a malignant case) are critical. This repository acts as a model selection testing ground. It imports a single clinical dataset, routes it through an automated preprocessing pipeline, evaluates 6+ classical machine learning architectures, and identifies the absolute best-performing algorithm based on precision, recall, and overall accuracy.

---

## 📊 Dataset Architecture (`Data.csv`)

The model utilizes clinical data derived from fine-needle aspirate (FNA) samples. The dataset contains 10 cytological attributes scored on a standardized scale from 1 to 10, alongside a categorical target variable.

| Feature Column | Attribute Description |
| :--- | :--- |
| `Sample code number` | Unique patient identification index (dropped during training) |
| `Clump Thickness` | Aggregation of epithelial cells in groupings |
| `Uniformity of Cell Size` | Consistency of cell growth dimensions |
| `Uniformity of Cell Shape` | Identification of irregular or distorted cell margins |
| `Marginal Adhesion` | Tendency of cells to retain structural unity rather than break apart |
| `Single Epithelial Cell Size` | Evaluation of significantly enlarged epithelial cell mass |
| `Bare Nuclei` | Percentage of nuclei isolated completely without surrounding cytoplasm |
| `Bland Chromatin` | Textural classification of nuclear material uniformity |
| `Normal Nucleoli` | Structure, visibility, and size of nucleoli inside the nucleus |
| `Mitoses` | Rate of active cell reproduction and nuclear division |
| **`Class`** | **Target Label:** `2` for Benign, `4` for Malignant |

---

## 🛠️ Data Preprocessing Pipeline

To ensure maximum model performance and prevent data leakage, the following pipeline is executed sequentially:

1. **Feature Extraction:** The `Sample code number` column is parsed out from the input matrix `X` since it contains zero predictive value.
2. **Missing Data Imputation:** Empty values or missing numbers (such as empty cells in the `Bare Nuclei` column) are intercepted and dynamically imputed using `scikit-learn`'s `SimpleImputer` configured with a column-wise `mean` strategy.
3. **Dataset Splitting:** The data is split into a Training set (typically 75-80%) to build model weights and a Test set (20-25%) to evaluate real-world generalization capability.
4. **Feature Scaling:** Since columns use relative scales or distances, `StandardScaler` is applied across all independent variables to guarantee distance-based models (like KNN or SVM) perform optimally.

---

## 🤖 Models Evaluated & Compared

The pipeline runs a comparative matrix across the following supervised learning classification models:

* **Logistic Regression:** Linear probabilistic boundary classification.
* **K-Nearest Neighbors (K-NN):** Multi
