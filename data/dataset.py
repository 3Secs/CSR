import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from .image_loader import ImageLoader
from .json_processor import JSONParser


class MRIDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.image_files = ImageLoader.get_image_files(image_dir)
        self.valid_files, self.invalid_files = self._filter_valid_files()

        if self.invalid_files:
            print(f"\n[Warning] {len(self.invalid_files)} invalid files skipped.")
            # 打印前5个无效文件示例
            for f in self.invalid_files[:5]:
                print(f"  - {f}")
            if len(self.invalid_files) > 5:
                print(f"  ... (total {len(self.invalid_files)} invalid files)")

    def _filter_valid_files(self):
        valid_files = []
        invalid_files = []
        for img_path in self.image_files:
            json_name = os.path.splitext(os.path.basename(img_path))[0] + '.json'
            json_path = os.path.join(self.json_dir, json_name)
            if not os.path.exists(json_path):
                invalid_files.append(f"{img_path} (JSON not found)")
                continue
            try:
                with open(json_path) as f:
                    data = json.load(f)
                    if ('annotations' in data and len(data['annotations']) > 0) or \
                            ('shapes' in data and len(data['shapes']) > 0):
                        valid_files.append(img_path)
                    else:
                        invalid_files.append(f"{json_path} (empty annotations)")
            except Exception as e:
                invalid_files.append(f"{json_path} (parse error: {str(e)})")
        return valid_files, invalid_files

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        image_path = self.valid_files[idx]
        try:
            # 加载图像
            image = ImageLoader.load_image(image_path)
            # 加载标注
            json_name = os.path.splitext(os.path.basename(image_path))[0] + '.json'
            json_path = os.path.join(self.json_dir, json_name)
            parser = JSONParser(json_path, image.shape[:2])
            mask = parser.get_mask()
            # 应用数据增强
            # 应用数据增强
            if self.transform:
                image_transform, mask_transform = self.transform  # 解包两个transform
                image = image_transform(image)
                mask = mask_transform(mask)

            return image, mask
        except Exception as e:
            print(f"\n[Error] Processing {image_path} failed: {str(e)}")
            # 返回占位数据保持批次完整性
            return torch.zeros(1, 640, 640), torch.zeros(1, 640, 640)


# dataset.py 修改后的 get_transforms 函数
def get_transforms(img_size=640):
    """返回图像和掩码的独立处理流程"""
    # 图像增强（包含标准化）
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 图像归一化到[-1,1]
    ])

    # 掩码处理（仅调整尺寸和转Tensor）
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()  # 自动归一化到[0,1]
    ])

    return image_transform, mask_transform  # 返回两个独立的transform


def get_dataloaders(config):
    """生成优化的数据加载器"""
    # 初始化数据集
    full_dataset = MRIDataset(
        image_dir=config["data"]["image_dir"],
        json_dir=config["data"]["json_dir"],
        transform=None  # 延迟应用transform以节省内存
    )

    # 划分训练集和验证集
    train_files, val_files = train_test_split(
        full_dataset.valid_files,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # 定义双transform
    image_transform, mask_transform = get_transforms(config["data"]["img_size"])
    transform = (image_transform, mask_transform)  # 打包成元组

    # 创建子数据集时传递双transform
    train_dataset = MRIDataset(
        image_dir=config["data"]["image_dir"],
        json_dir=config["data"]["json_dir"],
        transform=transform  # 传递元组
    )
    train_dataset.valid_files = train_files

    val_dataset = MRIDataset(
        image_dir=config["data"]["image_dir"],
        json_dir=config["data"]["json_dir"],
        transform=transform
    )
    val_dataset.valid_files = val_files

    # 配置高性能DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=12,  # 根据CPU逻辑核心数调整（i5-12400F为12线程）
        pin_memory=True,  # 启用锁页内存加速GPU传输
        persistent_workers=True,  # 保持子进程存活
        prefetch_factor=2  # 预取2个批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=8,  # 验证集可减少工作线程
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader