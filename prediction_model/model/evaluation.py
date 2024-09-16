from sklearn import metrics
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

# Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# Model evaluation metrics
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')

    # Ensure the directory exists
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    plot_path = plot_dir / "ROC_curve.png"
    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")  # Debugging line

    # Close plot
    plt.close()

    return accuracy, f1, auc
