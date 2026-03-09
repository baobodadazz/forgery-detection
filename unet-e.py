# train_resnet50_unet_with_val.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, jaccard_score
import segmentation_models_pytorch as smp
from tqdm import tqdm
import random

# ----------------------------
# 配置
# ----------------------------
TRAIN_DATA_DIR = './recodai-luc-scientific-image-forgery-detection/train_images'
MASK_DIR = './recodai-luc-scientific-image-forgery-detection/train_masks'
BATCH_SIZE = 8
IMG_SIZE = 256
EPOCHS = 30
VAL_RATIO = 0.1  # 从训练集中分出 10% 作为验证集
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# 自定义数据集（支持 train / val）
# ----------------------------

class ForgeryDataset(Dataset):
    def __init__(self, auth_paths, forge_paths, mask_dir, img_size=256):
        self.img_size = img_size
        self.auth_paths = auth_paths
        self.forge_paths = forge_paths
        self.mask_dir = mask_dir
        self.all_paths = self.auth_paths + self.forge_paths
        self.is_forged = [False] * len(self.auth_paths) + [True] * len(self.forge_paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        img_path = self.all_paths[idx]
        is_forged = self.is_forged[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 构建掩码
        if is_forged:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_dir, base_name + '.npy')
            mask = np.load(mask_path)

            mask = cv2.resize(mask[0], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            # 确保二值化（0 或 1）
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        img_tensor = self.transform(img)  # [3, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor

# ----------------------------
# 损失函数
# ----------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

# ----------------------------
# 验证函数
# ----------------------------
def validate(model, dataloader, device):
    model.eval()
    ious = []
    dices = []
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Validating", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)  # [B, 1, H, W]

            outputs = torch.sigmoid(model(imgs))
            preds = (outputs > 0.5).float()  # 二值化预测

            # 转为 numpy 用于 sklearn
            preds_np = preds.cpu().numpy().flatten()
            masks_np = masks.cpu().numpy().flatten()

            # 忽略全零掩码样本（真实无篡改），否则 IoU=0 会拉低平均值
            if masks_np.sum() == 0 and preds_np.sum() == 0:
                iou = 1.0
                dice = 1.0
            elif masks_np.sum() == 0 or preds_np.sum() == 0:
                iou = 0.0
                dice = 0.0
            else:
                iou = jaccard_score(masks_np, preds_np, zero_division=0)
                dice = f1_score(masks_np, preds_np, zero_division=0)

            ious.append(iou)
            dices.append(dice)

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    return mean_iou, mean_dice

# ----------------------------
# 训练主函数
# ----------------------------
def main():
    # 获取所有图像路径
    auth_imgs = sorted([os.path.join(TRAIN_DATA_DIR, 'authentic', f) 
                        for f in os.listdir(os.path.join(TRAIN_DATA_DIR, 'authentic')) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    forge_imgs = sorted([os.path.join(TRAIN_DATA_DIR, 'forged', f) 
                         for f in os.listdir(os.path.join(TRAIN_DATA_DIR, 'forged')) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 随机打乱并划分验证集
    random.seed(42)
    random.shuffle(auth_imgs)
    random.shuffle(forge_imgs)

    n_val_auth = max(1, int(len(auth_imgs) * VAL_RATIO))
    n_val_forge = max(1, int(len(forge_imgs) * VAL_RATIO))

    val_auth = auth_imgs[:n_val_auth]
    train_auth = auth_imgs[n_val_auth:]
    val_forge = forge_imgs[:n_val_forge]
    train_forge = forge_imgs[n_val_forge:]

    # 创建数据集
    train_dataset = ForgeryDataset(train_auth, train_forge, MASK_DIR, IMG_SIZE)
    val_dataset = ForgeryDataset(val_auth, val_forge, MASK_DIR, IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 模型
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",  # 若仍有 HF 问题，可设为 None
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_iou = 0.0

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = torch.sigmoid(model(imgs))
            loss = bce_dice_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        # 验证
        val_iou, val_dice = validate(model, val_loader, DEVICE)
        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | "
              f"Val Dice: {val_dice:.4f}")

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_model.pth")
            print(f"→ Saved best model with IoU: {best_iou:.4f}")

        scheduler.step()

    print("Training finished!")

if __name__ == "__main__":
    main()