# train_resnet50_unet.py
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

# 配置
DATA_DIR = './recodai-luc-scientific-image-forgery-detection/train_images'
MASK_DIR = './recodai-luc-scientific-image-forgery-detection/train_masks'
BATCH_SIZE = 16
IMG_SIZE = 512
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 自定义数据集
class ForgeryDataset(Dataset):
    def __init__(self, data_dir, mask_dir, img_size=256):
        self.img_size = img_size    # 统一图像大小
        self.auth_paths = [os.path.join(data_dir, 'authentic', f) for f in sorted(os.listdir(os.path.join(data_dir, 'authentic')))] # 真实图像路径列表
        self.forge_paths = [os.path.join(data_dir, 'forged', f) for f in sorted(os.listdir(os.path.join(data_dir, 'forged')))]  # 伪造图像路径列表
        self.mask_dir = mask_dir    # 掩码目录路径
        self.all_paths = self.auth_paths + self.forge_paths # 所有图像路径列表
        self.is_forged = [False] * len(self.auth_paths) + [True] * len(self.forge_paths)    # 图像是否伪造的标签列表

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
            raise FileNotFoundError(f"Image not found: {img_path}") # 没图像，抛异常
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 构建掩码
        if is_forged:# 如果是伪造图像，加载对应的掩码
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_dir, base_name + '.npy')
            mask = np.load(mask_path)

            mask = cv2.resize(mask[0], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            # 确保二值化（0 或 1）
            mask = (mask > 0).astype(np.float32)    # mask > 0 返回一个布尔列表，转换成 float32 类型，相当于将其二值化
        else:   # 不是伪造图像，全黑掩码
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        img_tensor = self.transform(img)  # [3, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor

# 损失函数：BCE（二元交叉熵） + Dice
# Dice 损失 就是将 Dice系数反过来当损失用， Dice系数 0~1 越大表示预测越好，Dice损失=1-Dice系数
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()    # 其实这个是内存整理，不整理的话，一些操作会导致物理顺序与逻辑顺序不一致，pytorch用stride（步长）来做映射，不复制数据，contiguous会让数据在内存中按逻辑顺序重新排列
    target = target.contiguous()    # 因为sum等操作需要调用底层cuda/mkl接口，要求输入张量物理地址连续，调用contiguous确保内存顺序正确
    intersection = (pred * target).sum(dim=(2, 3))      # 在H和W维度上求和，得到每个样本的交集
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)  # 计算每个样本的 Dice 系数
    return 1 - dice.mean()

def bce_dice_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

# ----------------------------
# 训练函数
# ----------------------------
def train():
    # 数据集 & 加载器
    dataset = ForgeryDataset(DATA_DIR, MASK_DIR, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 模型：ResNet50 + U-Net
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",  # 可改为 None（通义说生物医学图像建议微调或从头训）
        in_channels=3,
        classes=1,
        activation=None  # 不加 sigmoid，loss 内部处理
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = torch.sigmoid(model(imgs))  # [B, 1, H, W]
            loss = bce_dice_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")
        scheduler.step()

        # 保存模型
        torch.save(model.state_dict(), f"resnet18_unet_epoch_{epoch+1}.pth")

    print("Training finished!")

if __name__ == "__main__":
    train()