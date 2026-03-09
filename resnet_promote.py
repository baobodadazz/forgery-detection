import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast  # 混合精度支持

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import transforms

# ----------------------------
# 设备与路径设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BASE_DIR = pathlib.Path("./recodai-luc-scientific-image-forgery-detection")
TRAIN_IMG_DIR = BASE_DIR / "train_images"
AUTHENTIC_DIR = TRAIN_IMG_DIR / "authentic"
FORGED_DIR = TRAIN_IMG_DIR / "forged"

print("Number of authentic images:", len(list(AUTHENTIC_DIR.iterdir())))
print("Number of forged images:", len(list(FORGED_DIR.iterdir())))

# 构建 DataFrame
IMG_PATHS = []
for p in AUTHENTIC_DIR.iterdir():
    IMG_PATHS.append({"type": "authentic", "path": p})
for p in FORGED_DIR.iterdir():
    IMG_PATHS.append({"type": "forged", "path": p})

df = pd.DataFrame(IMG_PATHS).sample(frac=1, random_state=42).reset_index(drop=True)
df["label"] = df["type"].map({"authentic": 0, "forged": 1})

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print("Train size:", train_df.shape[0], "Test size:", test_df.shape[0])

# ----------------------------
# Dataset & DataLoader
# ----------------------------
class LUCDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row.label

    def __len__(self):
        return len(self.df)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = LUCDataset(train_df, transform)
test_ds = LUCDataset(test_df, transform)

# 优化 DataLoader（注意：Windows 用户可能需将 num_workers 设为 0）
num_workers = 4 if device.type == 'cuda' else 0
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=num_workers)
test_dl = DataLoader(test_ds, batch_size=32, pin_memory=True, num_workers=num_workers)

# ----------------------------
# Model Definition
# ----------------------------
class BASEModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# 分阶段训练函数
# ----------------------------
def train_model(model, train_dl, test_dl, phase_name, epochs, lr, freeze_backbone=True):
    print(f"\n=== {phase_name} ===")
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        # 只训练新添加的 fc 层
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        # 解冻全部或部分层（这里解冻 layer4）
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    scaler = GradScaler()
    best_val_loss = float('inf')

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for img, labels in train_dl:
            img, labels = img.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                pred = model(img)
                loss = nn.CrossEntropyLoss()(pred, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_losses.append(loss.item())
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for img, labels in test_dl:
                img, labels = img.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast():
                    out = model(img)
                    loss = nn.CrossEntropyLoss()(out, labels)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        test_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"./best_model_{phase_name.lower().replace(' ', '_')}.pth")
            print(f"✅ New best model saved at epoch {epoch+1} (val loss: {avg_val_loss:.4f})")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, test_losses

# ----------------------------
# 启动训练
# ----------------------------
model = BASEModel().to(device)

# 阶段1：仅训练分类头
train_losses1, test_losses1 = train_model(
    model, train_dl, test_dl,
    phase_name="Phase 1: Train Head Only",
    epochs=10,
    lr=2e-4,
    freeze_backbone=True
)

# 阶段2：微调 layer4 + 分类头
train_losses2, test_losses2 = train_model(
    model, train_dl, test_dl,
    phase_name="Phase 2: Fine-tune Layer4 + Head",
    epochs=10,
    lr=2e-5,
    freeze_backbone=False
)

# 合并损失用于绘图（可选）
total_train_losses = train_losses1 + train_losses2
total_test_losses = test_losses1 + test_losses2

# ----------------------------
# 最终评估
# ----------------------------
model.load_state_dict(torch.load("./best_model_phase_2_fine-tune_layer4_+_head.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for img, labels in test_dl:
        img = img.to(device, non_blocking=True)
        with autocast():
            out = model(img)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
print("\n🎯 Final Test Accuracy:", acc)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Authentic", "Forged"],
            yticklabels=["Authentic", "Forged"])
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# 生成提交文件
# ----------------------------
submission = pd.DataFrame({
    "Id": test_df.index,
    "Predicted": all_preds
})
submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print("✅ submission.csv saved at:", submission_path)