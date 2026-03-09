# 图像伪造检测项目 (Sample Forgery Detection)

这是一个基于深度学习的图像伪造检测项目，使用U-Net进行像素级分割检测伪造区域，并使用ResNet进行图像级分类判断图像是否为伪造。

## 项目结构

```
sample-forgery-detection/
├── environment.yml                    # Conda环境配置文件
├── npy2png.py                         # 将.npy掩码文件转换为PNG图像的工具脚本
├── resnet_promote.py                  # ResNet模型提升脚本
├── resnet.py                          # ResNet分类模型训练脚本
├── Unet_inference.py                  # U-Net推理脚本
├── Unet_train.py                      # U-Net训练脚本
├── unet-e.py                          # U-Net训练脚本（带验证集）W
```

## 环境配置

1. 安装Conda（如果尚未安装）
2. 创建环境：
   ```bash
   conda env create -f environment.yml
   ```
3. 激活环境：
   ```bash
   conda activate DINVMark_env
   ```

## 数据集

数据集放置于 `recodai-luc-scientific-image-forgery-detection/` 文件夹中：

- `train_images/authentic/`：真实图像
- `train_images/forged/`：伪造图像
- `train_masks/`：对应的伪造区域掩码（.npy格式）
- `test_images/`：测试图像
- `supplemental_images/` 和 `supplemental_masks/`：补充数据

## 模型和脚本说明

### U-Net分割模型

- **训练脚本**：
  - `Unet_train.py`：使用ResNet50作为编码器的U-Net训练脚本
  - `unet-e.py`：带验证集的U-Net训练脚本

- **推理脚本**：
  - `Unet_inference.py`：对单张图像或文件夹进行推理，生成伪造区域掩码

- **模型权重**：
  - 训练好的模型权重已经删除，项目十分简单可自行研究

### 工具脚本

- `npy2png.py`：将.npy格式的掩码转换为PNG图像

### 进行推理

对单张图像进行推理：

```python
from Unet_inference import predict_image
predict_image('path/to/model.pth', 'path/to/image.jpg')
```

对文件夹进行推理：

```python
from Unet_inference import predict_folder
predict_folder('path/to/model.pth', 'path/to/input_folder', 'path/to/output_folder')
```

## 依赖项

主要依赖包包括：

- PyTorch
- Torchvision
- Segmentation Models PyTorch (smp)
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm

所有依赖已在 `environment.yml` 中列出。

## 结果

- 训练好的模型权重保存在相应文件夹中
- 推理结果保存在 `predicted_masks/` 文件夹中
- `submission.csv` 包含最终的提交结果

## 注意事项

- 确保CUDA可用以加速训练和推理
- 图像大小统一为512x512或256x256（根据脚本配置）
- 掩码为二值图像，255表示伪造区域，0表示真实区域

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

使用GPL协议开源