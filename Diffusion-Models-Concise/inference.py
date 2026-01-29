import torch
from modules import UNet
from ddpm import Diffusion
from utils import save_images

def generate_new_images():
    # 1. 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 64
    model_path = "models/DDPM_Uncondtional/ckpt.pt"  # 训练好的模型路径
    
    # 2. 初始化模型结构 
    model = UNet(device=device).to(device)
    
    # 3. 加载训练好的权重
    print(f"正在加载模型: {model_path}")
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    
    # 4. 初始化扩散过程
    diffusion = Diffusion(img_size=img_size, device=device)
    
    # 5. 开始生成！(这里我们让它一次生成 16 张图)
    print("正在生成图片")
    # n=16 表示生成16张，模型会从纯噪声一步步去噪
    x = diffusion.sample(model, n=16)
    
    # 6. 保存结果
    save_images(x, "final_result.jpg")

if __name__ == "__main__":
    generate_new_images()