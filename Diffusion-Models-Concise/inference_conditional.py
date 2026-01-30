import torch
import os
import copy
from ddpm_conditional import Diffusion
from modules import UNet_conditional, EMA
from torchvision.utils import save_image, make_grid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 64
NUM_CLASSES = 10

MODEL_PATH = "models/DDPM_conditional"

NUM_IMAGES_PER_CLASS = 4  # 每个类别生成几张图
CFG_SCALE = 3.0           # 引导系数：越大越听话，越小越有创意
USE_EMA = True            # 指数移动平均 -> 图像质量通常更好

def run_inference():
    # 创建 diffusion（只负责噪声调度与采样过程）
    diff = Diffusion(img_size=IMG_SIZE, device=DEVICE)

    # 创建模型与 EMA model
    model = UNet_conditional(num_classes=NUM_CLASSES).to(DEVICE)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # 加载权重
    print(f"从 {MODEL_PATH} 加载权重...")
    ckpt_path = os.path.join(MODEL_PATH, "ckpt.pt")
    ema_path = os.path.join(MODEL_PATH, "ema_ckpt.pt")
    if not os.path.exists(ckpt_path):
        print(f"找不到模型文件: {ckpt_path}")
        return

    try:
        model_sd = torch.load(ckpt_path, map_location=DEVICE)
        # 有的 checkpoint 直接是 state_dict，也有可能是包含键的 dict，容错处理
        if isinstance(model_sd, dict) and "model" in model_sd:
            model.load_state_dict(model_sd["model"])
        else:
            model.load_state_dict(model_sd)
        print("模型权重加载成功 (ckpt).")
    except Exception as e:
        print(f"加载 ckpt 失败: {e}")
        return

    if os.path.exists(ema_path):
        try:
            ema_sd = torch.load(ema_path, map_location=DEVICE)
            if isinstance(ema_sd, dict) and "model" in ema_sd:
                ema_model.load_state_dict(ema_sd["model"])
            else:
                ema_model.load_state_dict(ema_sd)
            print("EMA 权重加载成功 (ema_ckpt).")
        except Exception as e:
            print(f"加载 ema_ckpt 失败: {e}")
            # 不阻塞，继续使用普通 model

    # 生成标签（每类多张）
    labels = []
    for i in range(NUM_CLASSES):
        labels.extend([i] * NUM_IMAGES_PER_CLASS)
    labels = torch.tensor(labels).to(DEVICE).long()
    n = labels.shape[0]
    print(f"正在生成 {n} 张图片 (每类 {NUM_IMAGES_PER_CLASS} 张)...")

    model_to_use = ema_model if (USE_EMA and os.path.exists(ema_path)) else model

    # 调用 concise 版 Diffusion.sample(model, n, labels, cfg_scale=...)
    generated_images = diff.sample(model_to_use, n=n, labels=labels, cfg_scale=CFG_SCALE)

    # sample 返回 uint8 0..255，转换为 tensor 保持 uint8
    # 我们需要把 batch 重新排序成按列是类别的布局，然后以 nrow=NUM_CLASSES 生成 4x10 的网格
    R = NUM_IMAGES_PER_CLASS
    C = NUM_CLASSES
    # 生成重排索引：对于每一行 r，按类别 c 取图片 index = c*R + r
    order = [c * R + r for r in range(R) for c in range(C)]
    generated_images = generated_images[order]  # 重新排序后长度仍为 R*C

    # 生成网格，列数设为类别数 -> 每行包含 C 张图，行数为 R (4)
    grid = make_grid(generated_images, nrow=C, padding=2, normalize=False)  # [3, H, W]

    # 为图片添加列标签（CIFAR-10 类别）
    cifar_labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    # 将 grid 转为 numpy 图并用 matplotlib 输出带列标签的图片
    import matplotlib.pyplot as plt
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # H x W x C
    H, W, _ = grid_np.shape
    cell_w = IMG_SIZE + 2  # 单张图宽度 + padding
    centers_x = [(c * (IMG_SIZE + 2) + IMG_SIZE / 2) for c in range(C)]

    plt.figure(figsize=(C, R))
    plt.imshow(grid_np)
    ax = plt.gca()
    ax.set_xticks(centers_x)
    ax.set_xticklabels(cifar_labels, fontsize=8, rotation=90)
    ax.set_yticks([])
    plt.subplots_adjust(bottom=0.25)
    output_filename = "generated_result.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"结果已保存为: {os.path.abspath(output_filename)}")
if __name__ == "__main__":
    run_inference()