
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 临时绕过 OpenMP 冲突

import matplotlib
matplotlib.use('TkAgg')  # 强制使用稳定后端

import matplotlib.pyplot as plt


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.version.cuda)
print(f"Using device: {device}")

# --- 1. 参数配置 ---
# 核心参数
content_img_path = "img/Capture001.png" # 替换你的内容图路径
style_img_path = "img/img_1.png"    # 替换你的风格图路径
image_size = 512   # 统一处理的图片尺寸
num_steps = 1000   # 迭代次数
style_weight = 1e6 # 风格权重 β
content_weight = 1 # 内容权重 αj

# 用于提取特征的层
content_layers = ['conv_4']      # 内容特征层
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] # 风格特征层

# --- 2. 辅助函数 ---
def load_image(img_path, max_size=512, shape=None):
    """加载并预处理图像"""
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    image = in_transform(image)[:3 ,: ,:].unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    """将张量转换回可显示的图片"""
    image = tensor.cpu().clone().detach().squeeze(0)
    image = image.numpy().transpose(1 ,2 ,0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

def get_features(image, model, layers=None):
    """通过模型前向传播，提取指定层的特征"""
    if layers is None:
        layers = {'0': 'conv_1',
                  '5': 'conv_2',
                  '10': 'conv_3',
                  '19': 'conv_4',
                  '28': 'conv_5'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """计算Gram矩阵"""
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# --- 3. 主程序 ---
# 加载图像并打印
content = load_image(content_img_path, max_size=image_size)
style = load_image(style_img_path, shape=[content.size(2), content.size(3)])
# 将待优化图像初始化为内容图像副本
target = content.clone().requires_grad_(True).to(device)

# 加载预训练模型并切换至评估模式
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
# 冻结模型参数
for param in vgg.parameters():
    param.requires_grad_(False)

# 获取特征
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
# 计算风格图像的Gram矩阵
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# 定义优化器
optimizer = optim.Adam([target], lr=0.003)

# 主循环
for ii in range(1, num_steps +1):
    # 获取目标图像特征
    target_features = get_features(target, vgg)
    # 计算内容损失
    content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4'] )**2)
    # 计算风格损失
    style_loss = 0
    for layer in style_features:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean((target_gram - style_gram )**2)
        style_loss += layer_style_loss / len(style_features)
    # 总损失
    total_loss = content_weight * content_loss + style_weight * style_loss
    # 反向传播和优化
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 打印进度
    if ii % 100 == 0:
        print(f'Step [{ii}/{num_steps}], Total Loss: {total_loss.item():.4f}')
        # 可选：展示中间结果
        plt.imshow(im_convert(target))
        plt.title(f'Step {ii}')
        plt.show()