# Bank Term Deposit Prediction using SVM (Imbalanced Classification)

This project builds a machine learning pipeline to predict whether a customer will subscribe to a bank term deposit.

The dataset is highly imbalanced, so the main focus is on improving minority class performance using:

- Support Vector Machine (RBF Kernel)
- Class Weight Tuning
- Decision Threshold Optimization
- F1-score Maximization

---

## Dataset

Binary classification problem:

- Class 0: No deposit  
- Class 1: Deposit (minority class)

Distribution:

- Class 0: ~39,900 samples  
- Class 1: ~5,300 samples  

Because of this imbalance, accuracy is misleading. F1-score is used as the main metric.

---

## Objective

Build a classifier that maximizes F1-score for the minority class while maintaining reasonable overall accuracy.

---

## Tech Stack

- Python  
- scikit-learn  
- NumPy  
- Jupyter Notebook  

Techniques used:

- SVC (RBF Kernel)
- Pipeline + StandardScaler
- Class Weight Tuning
- Threshold Optimization
- Precision–Recall Analysis

---

## Workflow

### Baseline Model

Initial SVM without class weighting:

- High accuracy
- Very poor recall for minority class
- Minority F1 ≈ 0.43

This confirms the imbalance problem.

---

### Class Weight Tuning

Manual testing:

| Weight | F1 |
|--------|----|
| 2 | 0.597 |
| 3 | 0.601 |
| 4 | 0.595 |
| 5 | 0.589 |

Best result:

class_weight = {0:1, 1:3}


---

### Threshold Optimization

Instead of default prediction threshold:

- Decision scores extracted
- Precision–Recall curve computed
- F1 calculated for each threshold
- Best threshold selected via argmax(F1)

---

### Final Pipeline

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", class_weight={0:1, 1:3}))
])
Final Results
Class 0: Precision 0.96 | Recall 0.91 | F1 0.94
Class 1: Precision 0.52 | Recall 0.72 | F1 0.60

Accuracy: 0.89
Final Minority F1: 0.6006
Key Takeaways
Accuracy alone is unreliable for imbalanced data.

Class weighting significantly improves recall.

Threshold tuning provides additional F1 gains.

SVM performs well on tabular data with proper scaling.

Future Improvements
XGBoost / LightGBM

SMOTE

SHAP explainability

Nested cross-validation

Gamma + C hyperparameter tuning

How to Run
pip install -r requirements.txt
jupyter notebook
Author
Hasan Badalli
Machine Learning / Software Engineering