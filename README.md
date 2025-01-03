# EE353_IITB

## Overview

This repository contains the assignments for the **EE353** and **EE769** courses at **IIT Bombay**, focusing on **Linear Models**, **Regularization**, **Clustering**, and **Principal Component Analysis (PCA)**. The objective of these assignments is to provide hands-on experience in regression analysis, regularization techniques, classification models, and unsupervised learning.

---

## Assignment Overview

### 1. **Data Generation for Regression**

- **Objective**: Implement functions to generate data matrices for regression and target vectors with added noise.
- **Outcome**: Developed a function to generate an input data matrix `X` of size `NxD` and target vector `t` of size `Nx1`, based on a weight vector `w` and noise variance `σ`.

### 2. **Gradient Functions for Regularization**

- **Objective**: Implement gradient functions for Mean Squared Error (MSE), L1, and L2 regularization, and use these functions in gradient descent.
- **Outcome**: Applied the gradients to optimize regression models with regularization, specifically for L1 and L2 norms.

### 3. **Impact of Noise, Data Size, and Regularization**

- **Objective**: Conduct experiments to examine the effect of noise variance, data size, and regularization parameters on model accuracy.
- **Outcome**: Analyzed how different values of noise variance and dataset size impact the accuracy of linear regression models, and fine-tuned regularization parameters for improved performance.

### 4. **Elastic Net and Variable Elimination**

- **Objective**: Explore the impact of Elastic Net regularization and variable elimination on correlated data columns.
- **Outcome**: Investigated how regularization techniques help eliminate variables and how Elastic Net behaves with correlated features.

### 5. **Binary Classification with Linear Regression**

- **Objective**: Apply linear regression to binary classification and implement logistic regression optimization techniques.
- **Outcome**: Optimized a binary classification model using logistic regression and tested its performance.

### 6. **Clustering Customer Behavior**

- **Objective**: Perform data preprocessing and clustering (k-means, DBSCAN) to segment customer behavior.
- **Outcome**: Conducted customer segmentation using clustering techniques to identify behavioral patterns.

### 7. **Principal Component Analysis (PCA)**

- **Objective**: Use PCA to identify key components explaining customer data variance and reconstruct the data for analysis.
- **Outcome**: Applied PCA to extract significant features from customer data and visualize the data reconstruction.

---

## Requirements

To run the code for these assignments, you will need:

- Python 3.x
- Libraries: `numpy`, `matplotlib`, `sklearn`, `pandas`

You can install the necessary libraries using the following command:

```bash
pip install numpy matplotlib scikit-learn pandas
```

