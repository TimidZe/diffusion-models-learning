# Learning Path
First, we studied several important modules in Stable-Diffusion, including VAE, CLIP, UNet, and DDPM(*). These modules are all implemented in the Stable-Diffusion folder. Because DDPM uses skipping steps for speedup, resulting in poor performance, we learned DDIM instead. Second, since Stable-Diffusion uses pre-trained models, we needed to understand the complete training process.

# Project Structure

1. Theory: Classic DDPM, DDIM, VAE

2. Application: Using complete Diffusion for inference to obtain some results

3. Completing the entire process from training to prediction

4. Greater Control: ControlNet (Onging)

5. More efficient training, more advanced runtime management, domain-specific datasets (MedMNIST) (Onging)

# Project Implementation Description
## 1. Stable Diffusion （Inference）

![image](https://github.com/TimidZe/diffusion-models-learning/blob/main/IMG/%E6%B5%81%E7%A8%8B%E5%9B%BE1.png)

### 1.1 内容

(In order of serial number) Self/Cross Attention、VAE Encoder/Decoder、CLIP、UNet、Scheduler

### 1.2 Results
1. Noising Process

![image](https://github.com/TimidZe/diffusion-models-learning/blob/main/IMG/%E5%8A%A0%E5%99%AA%E5%9B%BE2.png)

2. Txt2Img

Config:
```
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 8k resolution."
uncond_prompt = ""
do_cfg = True
cfg_scale = 7
```
![image](https://github.com/TimidZe/diffusion-models-learning/blob/main/IMG/%E6%96%87%E7%94%9F%E5%9B%BE3.png)

3. Img2Img

![image](https://github.com/TimidZe/diffusion-models-learning/blob/main/IMG/result4.png)
bty This is my kitten, Coconut :)

### 1.3 Discussion

The reasons why img2img has bad results are as follows:
1. The Task Mismatch: Img2Img is not Inpainting. This project implements full-image repainting. To perfectly implement "covering with a blanket," inpainting (partial repainting) is needed, which requires additional mask input, and mask input is currently not supported.

2. The native model of Stable Diffusion v1.5 is a descriptive model, which has poor understanding of "instruction-type" prompts. It should describe the final image state.

## 2. Diffusion-Concise （Train & Inference）

### 2.1 Training Results

Training and inference of unconditional and conditional ddpm were performed respectively.

![image](https://github.com/TimidZe/diffusion-models-learning/blob/main/IMG/Training_results5.png)


![image](https://github.com/TimidZe/diffusion-models-learning/blob/main/IMG/120_ema.jpg)

Results of the 120th epoch of training

### 2.2 Inference Results

![image](https://github.com/TimidZe/diffusion-models-learning/blob/main/IMG/generated_result.png)

Inference Result by the ema model trained for xxx epochs 


## References
1. https://github.com/hkproj/pytorch-stable-diffusion
2. https://github.com/dome272/Diffusion-Models-pytorch
