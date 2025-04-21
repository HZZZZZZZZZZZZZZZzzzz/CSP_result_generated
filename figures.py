
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import os

# ========= Load Data =========
data_dir = r"D:\PSYCHOPY_WORKFLOW\real_time\csplda_model"
cv_scores = np.load(os.path.join(data_dir, "cv_scores.npy"))
X_csp = np.load(os.path.join(data_dir, "X_csp.npy"))
y_pred = np.load(os.path.join(data_dir, "y_pred.npy"))
y_true = np.load(os.path.join(data_dir, "y_true.npy"))


# ========= Output Directory =========
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# ========= 1. Cross-validation Accuracy =========
plt.figure()
plt.plot(cv_scores, marker='o')
plt.axhline(0.5, linestyle='--', color='r', label='Chance')
plt.title("Cross-validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "cv_accuracy.png"))
plt.close()

# ========= 2. Confusion Matrix =========
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["hands_left", "hands_right"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# ========= 3. CSP Feature Distribution (Component 1) =========
plt.figure()
plt.boxplot([X_csp[:25, 0], X_csp[25:, 0]], labels=["hands_left", "hands_right"])
plt.title("CSP Component 1 Feature Distribution (Test Set)")
plt.ylabel("Average Log Power")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "csp_feature_boxplot.png"))
plt.close()

# ========= 4. Per-Class Accuracy Bar Chart =========
class_accuracies = []
class_labels = ["hands_left", "hands_right"]
for cls in [0, 1]:
    idx = (y_true == cls)
    correct = (y_pred[idx] == cls).sum()
    acc = correct / idx.sum()
    class_accuracies.append(acc)

plt.figure()
plt.bar(class_labels, class_accuracies, color=["#1f77b4", "#ff7f0e"])
plt.ylim(0, 1)
plt.title("Per-Class Accuracy")
plt.ylabel("Accuracy")
plt.grid(axis="y")
plt.savefig(os.path.join(output_dir, "class_accuracy.png"))
plt.close()

print(f"✅ All 4 figures saved to {output_dir}")
