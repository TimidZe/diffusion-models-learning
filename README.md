# 学习路线
首先将Diffusion中几个重要的模块学习，包括VAE，CLIP, UNet, DDPM(*), 上述模块均在Stable-Diffusion文件夹中实现，由于DDPM中采用为了加速采用跳步导致效果不好，因此学习了DDIM；其次由于Stable-Diffusion是通过调用预训练模型，为了掌握完整的训练

# 项目构建思路
1. 理论：经典DDPM、DDIM、VAE
2. 应用：使用完整的Diffusion进行推理，获得一些结果
3. 补全从训练到预测的整个流程
4. 更可控：ControlNet （Onging）
5. 更高效的训练、更高级的运行管理、特定领域的数据集(MedMNIST) （Onging）

# 项目实现说明
## 1. Stable Diffusion （Inference）

【流程图1】

### 1.1 内容

（按序号顺序）自/交叉注意力、VAE Encoder/Decoder、CLIP、UNet、Scheduler

### 1.2 Results
1. 加噪过程

【加噪图2】

2. 文生图

Config:
```
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 8k resolution."
uncond_prompt = ""
do_cfg = True
cfg_scale = 7
```
【文生图3】

3. 图生图
【result4】
bty这是我的小猫椰子

### 1.3 Discussion

图生图效果不好的原因可能如下：
1. The Task Mismatch：Img2Img 不是 Inpainting。该项目实现的是全图重绘。要完美实现“盖毯子”，需要的是 Inpainting (局部重绘)，这需要额外的 Mask输入，而目前不支持 mask 输入。
2. Stable Diffusion v1.5 原生模型是描述型模型，对“指令型”提示词理解很差，应该描述最终画面状态

## 2. Diffusion-Concise （Train & Inference）

### 2.1 Training Results


## 参考资料
1. https://github.com/hkproj/pytorch-stable-diffusion
2. https://github.com/dome272/Diffusion-Models-pytorch
