# CSP-Based Motor Imagery Classification (EEG)

This repository demonstrates a simple pipeline for evaluating motor imagery classification performance using Common Spatial Pattern (CSP) features and standard machine learning techniques. It includes scripts to generate key performance visualisations based on preprocessed EEG data.

---

## 📂 Files Overview

| File | Description |
|------|-------------|
| `figures.py` | Python script to generate all visualisation figures |
| `cv_scores.npy` | Cross-validation accuracy across 5 folds |
| `X_csp.npy` | CSP features (log-variance, 1st component only) for 50 samples |
| `y_true.npy` | Ground-truth labels for two classes (hands_left = 0, hands_right = 1) |
| `y_pred.npy` | Model predictions corresponding to `y_true` |

---

## 📊 Visualisations

Running `figures.py` will generate four figures and save them into a `figures/` folder:

1. **Cross-validation Accuracy**  
   Line plot of classification accuracy across 5 folds (mean ≈ 70%).

2. **Confusion Matrix**  
   2×2 matrix showing true vs. predicted labels on the test set (overall accuracy ≈ 76%).

3. **CSP Feature Distribution**  
   Boxplot of log-variance values for the first CSP component across the two classes.

4. **Per-Class Accuracy**  
   Bar chart comparing classification accuracy for `hands_left` and `hands_right` samples.

---

## 🚀 How to Run

```bash
python figures.py
