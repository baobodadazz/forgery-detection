import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torchvision.models as models
from torchvision.models import ResNet18_Weights

# 换更大的resnet试试
from torchvision.models import ResNet50_Weights

from torchvision.transforms import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BASE_DIR = pathlib.Path("./recodai-luc-scientific-image-forgery-detection")
TRAIN_IMG_DIR = BASE_DIR / "train_images"   # 训练图像路径

AUTHENTIC_DIR = TRAIN_IMG_DIR / "authentic" # 真实图像路径（训练图像中）
FORGED_DIR = TRAIN_IMG_DIR / "forged"       # 伪造图像路径（训练图像中）

# 统计真是图像和伪造图像的数量
print("Number of authentic images:", len(list(AUTHENTIC_DIR.iterdir())))
print("Number of forged images:", len(list(FORGED_DIR.iterdir())))


IMG_PATHS = []

for p in AUTHENTIC_DIR.iterdir():
    IMG_PATHS.append({"type": "authentic", "path": p})
for p in FORGED_DIR.iterdir():
    IMG_PATHS.append({"type": "forged", "path": p})

df = pd.DataFrame(IMG_PATHS).sample(frac=1).reset_index(drop=True)  # 生成pandas的DataFrame，随机打乱顺序并重置索引
df["label"] = df["type"].map({"authentic": 0, "forged": 1})         # 将真实和伪造转换为数值标签

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)  # 将数据划分为测试集和训练集，stratify: 分层抽样，确保训练集和测试集在label的比例与原始数据集一致
print(train_df.shape)
print(test_df.shape)   # 输出训练集和测试集的大小

class LUCDataset(Dataset):
    # 定义一个名为 `LUCDataset` 的类，继承自 PyTorch 的 `torch.utils.data.Dataset`。
    # 所有自定义数据集都必须继承 `Dataset` 并实现 `__len__()` 和 `__getitem__()` 方法。
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform  # 可选的图像预处理操作（如缩放、归一化等）

    def __getitem__(self, idx):
        row = self.df.iloc[idx]     # 从df中提取第idx行
        img = Image.open(row.path).convert("RGB")   # 使用 PIL 打开图像，并转换为RGB格式
        if self.transform:                  # 如果 transform 不为None
            img = self.transform(img)       # 则对 img 进行 transform 变换
        return img, row.label               # 返回 图像 和 label

    def __len__(self):
        return len(self.df)                 # 返回总样本数

transform = transforms.Compose([            # 将多个图像变换组合成一个流水线
    transforms.Resize((224,224)),           # 统一所防止 224*224
    transforms.ToTensor(),                  # 将 PIL 图像（值范围 0–255）转换为 PyTorch Tensor（值范围 0.0–1.0），并把维度从 `(H, W, C)` 转为 `(C, H, W)`
    transforms.Normalize(                   # 对每个通道进行标准化（减均值、除标准差）（使用的是 ImageNet 数据集的统计值）
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

train_ds = LUCDataset(train_df, transform)      # 训练集的dataset
test_ds = LUCDataset(test_df, transform)        # 测试集的dataset

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)    # 生成训练集dataloader
test_dl = DataLoader(test_ds, batch_size=32)                    # 生成测试集dataloader

class BASEModel(nn.Module):
    def __init__(self, num_classes=2):  # num_classes 定义输出层，2为二分类
        super().__init__()
        # Offline ResNet18
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)   # 加载预定义resnet18架构，weights=None设置随机初始化权重（weights=ResNet18_Weights.IMAGENET1K_V1，使用ImageNet预训练权重）

        # Freeze backbone
        #for p in self.backbone.parameters():    # 冻结 resnet 主干网络的全部参数（因为没有加载预训练模型，这种做法通常不推荐）
        #    p.requires_grad = False

        # Small classifier for fast CPU
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

model = BASEModel().to(device)
print("Model initialized and moved to device.")
print("Model is now at " + str(next(model.parameters()).device))  # 检测模型参数所在设备
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

istrain = True  # 设置为 True 进行训练，False 则加载最优模型进行评估

if istrain:
    EPOCHS = 20
    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        #print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        batch_losses = []
        for img, labels in train_dl:
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, labels)
            #print("backward start")
            loss.backward()
            #print("optimizer step start")
            optimizer.step()

            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))

        # Test loss
        model.eval()
        test_batch_losses = []
        with torch.no_grad():
            for img, labels in test_dl:
                img, labels = img.to(device), labels.to(device)
                out = model(img)
                loss = loss_fn(out, labels)
                test_batch_losses.append(loss.item())
        test_losses.append(np.mean(test_batch_losses))

        # 保存最优模型
        if epoch == 0 or test_losses[-1] < min(test_losses[:-1]):
            torch.save(model.state_dict(), "./best_model.pth")
            print(f"Best model saved at epoch {epoch+1} with test loss: {test_losses[-1]:.4f}")

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_losses[-1]:.4f} - Test Loss: {test_losses[-1]:.4f}")
else:
    # 加载最优模型
    model.load_state_dict(torch.load("./best_model.pth"))
    print("Best model loaded for evaluation.")
'''




plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(test_losses, label="Test Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
'''
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for img, labels in test_dl:
        img = img.to(device)
        out = model(img)
        preds = torch.argmax(out, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
print("Test Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Authentic", "Forged"],
            yticklabels=["Authentic", "Forged"])
plt.title("Confusion Matrix")
plt.show()

# Submission File
submission = pd.DataFrame({
    "Id": test_df.index,
    "Predicted": all_preds
})
submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print("submission.csv saved at:", submission_path)

