import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils


class JSONParser:
    def __init__(self, json_path, img_size=(640, 640)):
        self.json_path = json_path
        self.img_size = img_size
        self.mask = np.zeros(img_size, dtype=np.uint8)

        # 加载 JSON 文件
        with open(json_path) as f:
            self.data = json.load(f)

        # 自动检测格式
        if 'annotations' in self.data:
            self.format = 'coco'
        elif 'shapes' in self.data:
            self.format = 'labelme'
        else:
            raise ValueError(f"Unsupported JSON format in {json_path}")

    def _parse_coco(self):
        """解析 COCO 格式的标注"""
        for ann in self.data['annotations']:
            if 'segmentation' in ann and isinstance(ann['segmentation'], dict):
                rle = ann['segmentation']
                binary_mask = maskUtils.decode(rle)
                self.mask = np.maximum(self.mask, binary_mask)
        return self.mask

    def _parse_labelme(self):
        """解析 Labelme 格式的标注"""
        for shape in self.data['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(self.mask, [points], color=255)
        return self.mask

    def get_mask(self):
        """统一接口生成掩码"""
        if self.format == 'coco':
            return self._parse_coco()
        elif self.format == 'labelme':
            return self._parse_labelme()
        else:
            raise ValueError("Unsupported format")
        # ...生成二值掩码的逻辑...
        mask = np.array(mask_image).astype(np.float32) / 255.0  # 归一化到[0,1]
        return torch.from_numpy(mask)