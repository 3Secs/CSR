import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(original, mask, prediction):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth")

    plt.subplot(133)
    plt.imshow(prediction, cmap='gray')
    plt.title("Prediction")

    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.close()


def plot_signal_distribution(image, mask):
    masked_region = image[mask > 0.5]
    plt.hist(masked_region.flatten(), bins=50, alpha=0.7)
    plt.xlabel("Signal Intensity")
    plt.ylabel("Frequency")
    plt.title("Signal Intensity Distribution")
    plt.savefig("histogram.png")
    plt.close()