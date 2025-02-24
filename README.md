# 基于U-Net的椎间盘信号量化系统

## 项目概述
本系统使用深度学习技术对腰椎MRI图像中的椎间盘区域进行自动分割，并量化椎间盘信号强度特征。系统特点：
- 端到端的U-Net分割模型
- 支持Dice/BCE混合损失函数
- 提供信号强度量化分析
- 可视化工具集成

## 安装指南
```bash
git clone https://github.com/yourrepo/unet_ivd_quant.git
cd unet_ivd_quant
pip install -r requirements.txt