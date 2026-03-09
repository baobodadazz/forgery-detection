import os
import numpy as np
from PIL import Image
import argparse

def convert_npy_to_png(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有 .npy 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            input_path = os.path.join(input_folder, filename)
            try:
                # 加载 .npy 文件
                data = np.load(input_path)

                # 检查形状是否为 (1, H, W)
                if data.ndim != 3:
                    print(f"跳过 {filename}：形状不符合 (1, H, W)，当前形状为 {data.shape}")
                    continue

                # 去掉第一个维度，得到 (H, W)
                img_array = data[0]

                # 可选：归一化到 [0, 255] 如果是浮点类型
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    # 假设数据范围是 [0, 1] 或需要自动缩放
                    img_min = img_array.min()
                    img_max = img_array.max()
                    if img_max > img_min:
                        img_array = (img_array - img_min) / (img_max - img_min) * 255
                    else:
                        img_array = np.zeros_like(img_array)
                # 转换为 uint8
                img_array = (img_array * 255).astype(np.uint8)
                # 构造输出路径
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)

                # 保存为 PNG
                img = Image.fromarray(img_array)
                img.save(output_path)

                print(f"已保存: {output_path}")

            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    input_dir = "./recodai-luc-scientific-image-forgery-detection/train_masks"
    output_dir = "./train_masks"

    convert_npy_to_png(input_dir, output_dir)