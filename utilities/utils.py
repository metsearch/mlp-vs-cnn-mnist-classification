import os

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from .log import logger
    
def display_images_with_predictions(path2metrics, arch, images, predictions, ground_truths, num_images=40, num_images_per_row=10):
    images_to_display = images[:num_images]
    num_rows = int(np.ceil(num_images / num_images_per_row))

    _, axes = plt.subplots(num_rows, num_images_per_row, figsize=(15, 3 * num_rows))

    for i in range(num_rows):
        for j in range(num_images_per_row):
            index = i * num_images_per_row + j
            if index < num_images:
                img = np.reshape(images_to_display[index], (28, 28))
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].set_title(f'{predictions[index]} ({ground_truths[index]})')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(path2metrics, f'{arch}_inference_examples.png'))
    plt.show()

def plot_confusion_matrix(path2metrics, arch, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(path2metrics, f'{arch}_confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':
    logger.info('Testing utils...')