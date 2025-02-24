import yaml
import torch
from torch import optim, nn
from torch.cuda import amp  # 新增混合精度支持
from data.dataset import get_dataloaders
from models.unet import UNet
from models.losses import BCEDiceLoss


def train(config_path):
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 初始化模型和优化器
    device = torch.device(config["training"]["device"])
    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        init_features=config["model"]["init_features"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = BCEDiceLoss()

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 加载数据
    train_loader, val_loader = get_dataloaders(config)

    # 训练循环
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)  # 异步传输
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 混合精度前向传播
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # 验证阶段
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        # 打印日志
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}\n")

    # 保存模型
    torch.save(model.state_dict(), "saved_models/best_model.pth")



if __name__ == "__main__":
    train("configs/base_config.yaml")