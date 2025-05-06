import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def log_confusion_matrix(y_true, y_pred, class_names, filename="confusion_matrix.png"):
    """
    Generates and saves a confusion matrix as an image.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticks=np.arange(len(class_names)) + 0.5,
                yticks=np.arange(len(class_names)) + 0.5)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=45)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_confusion_matrix_txt(cm, class_names, filename="confusion_matrix.txt"):
    """
    Saves the raw confusion matrix as a plain text file.
    """
    with open(filename, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write("Labels: " + str(class_names) + "\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
