import torch

def calculate_dice(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calculate_signal_intensity(image, mask):
    masked_region = image[mask > 0.5]
    mean = torch.mean(masked_region)
    std = torch.std(masked_region)
    return mean.item(), std.item()