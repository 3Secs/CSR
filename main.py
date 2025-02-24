import argparse

import cv2
import yaml
from scripts.train import train
from scripts.inference import Inferencer
from utils.visualize import plot_comparison
import sys
import os

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)


def main():
    parser = argparse.ArgumentParser(description="IVD Quantification System")
    parser.add_argument("--mode", choices=["train", "infer", "visualize"], required=True)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--image", help="Input image path")
    parser.add_argument("--model", help="Model path for inference")
    parser.add_argument("--mask", help="Ground truth mask path")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.config)
    elif args.mode == "infer":
        inferencer = Inferencer(args.config, args.model)
        pred_mask, mean, std = inferencer.predict(args.image)
        print(f"Quantification Results - Mean: {mean:.2f}, SD: {std:.2f}")
    elif args.mode == "visualize":
        original = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread("prediction.png", cv2.IMREAD_GRAYSCALE)
        plot_comparison(original, mask, pred)

if __name__ == "__main__":
    main()