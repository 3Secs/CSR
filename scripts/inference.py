import cv2
import torch
import yaml
import numpy as np
from models.unet import UNet
from utils.metrics import calculate_signal_intensity
from data.dataset import get_transforms


class Inferencer:
    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config["training"]["device"])
        self.model = self._load_model(model_path)
        self.transform = get_transforms(self.config["data"]["img_size"])

    def _load_model(self, model_path):
        model = UNet(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
            init_features=self.config["model"]["init_features"]
        ).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict(self, image_path):
        # 读取并预处理图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            pred_mask = self.model(tensor).cpu().numpy()[0][0]

        # 信号量化
        mean, std = calculate_signal_intensity(image, pred_mask)
        return pred_mask, mean, std


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

    inferencer = Inferencer(args.config, args.model)
    mask, mean, std = inferencer.predict(args.image)

    print(f"Mean Signal Intensity: {mean:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    cv2.imwrite("prediction.png", (mask * 255).astype(np.uint8))