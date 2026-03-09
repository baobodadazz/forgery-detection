# inference.py
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

# 推理
def infer_image(model, img_tensor):
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        mask = (pred[0, 0] > 0.5).numpy().astype(np.uint8) * 255
    return mask

# 加载模型和权重
def load_model(model_path):
    model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    return model

# 图像预处理
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    img = cv2.resize(img, (512, 512))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, H, W

# 推理一张图片
def predict_image(model_path, image_path):
    model = load_model(model_path)
    img_tensor, H, W = preprocess_image(image_path)
    mask = infer_image(model, img_tensor)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("predicted_mask.png", mask)

# 推理整个文件夹中的文件
def predict_folder(model_path, folder_path, output_folder="./predicted_masks"):
    import os
    model = load_model(model_path)
    count = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            img_tensor, H, W = preprocess_image(image_path)
            print(f"{count}:Processing {filename}...")
            mask = infer_image(model, img_tensor)
            count += 1
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            save_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_predicted_mask.png")
            cv2.imwrite(save_path, mask)

def main():
    # predict_image("./resnet50U/resnet18_unet_epoch_50.pth", "./recodai-luc-scientific-image-forgery-detection/test_images/45.png")
    predict_image("./18E/best_model.pth", "./recodai-luc-scientific-image-forgery-detection/train_images/forged/330.png")
    # predict_folder("./resnet50U/resnet18_unet_epoch_50.pth", "./recodai-luc-scientific-image-forgery-detection/train_images/forged")

if __name__ == "__main__":
    main()