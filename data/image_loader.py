import os
import cv2
import numpy as np


class ImageLoader:
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp')

    @classmethod
    def load_image(cls, path: str) -> np.ndarray:
        print(f"Attempting to load image: {path}")  # 调试输出
        """
        加载图像并统一为灰度图。

        参数:
            path (str): 图像文件路径。

        返回:
            np.ndarray: 灰度图像数组。

        异常:
            FileNotFoundError: 如果文件不存在。
            ValueError: 如果文件无法加载或格式不支持。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        if not path.lower().endswith(cls.SUPPORTED_FORMATS):
            raise ValueError(
                f"Unsupported image format. Supported formats: {cls.SUPPORTED_FORMATS}"
            )

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        print(f"Successfully loaded image: {path}")  # 调试输出
        return image

    @classmethod
    def get_image_files(cls, directory: str) -> list:
        """获取目录下所有支持的图像文件"""
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(cls.SUPPORTED_FORMATS)
        ]