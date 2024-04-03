# InstantID with FouriScale

We combined below two outstanding works for ID-Preserving high resolution image generation!

[InstantID: Zero-shot Identity-Preserving Generation in Seconds](https://github.com/InstantID/InstantID): Very 🐂🍺(oustanding) ID-Preserving generation model.

[FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](https://github.com/LeonHLJ/FouriScale): Solves the problem of repetitive patterns and structural distortions that occur when the model exceeds its trained resolution. 💯

# Usage

## Preparation

Clone this repository

```bash
git clone https://github.com/Norman-Ou/InstantID-with-FouriScale.git
```

Clone InstantID repository to the root of this repository

```bash
cd InstantID-with-FouriScale
git clone https://github.com/InstantID/InstantID.git
```

Download InstantID models 

1. Download the model from [Huggingface](https://huggingface.co/InstantX/InstantID). You also can download the model in python script:

   ```python
   from huggingface_hub import hf_hub_download
   hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./InstantID/checkpoints")
   hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./InstantID/checkpoints")
   hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./InstantID/checkpoints")
   ```

2. For face encoder, you need to manually download via this [URL](https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304) to `models/antelopev2` as the default link is invalid. Once you have prepared all models, the folder tree should be like:

   ```
     fouriscale
     InstantID
     ├── models
     ├── checkpoints
     ├── ip_adapter
     ├── ...
     ├── ...
     └── README.md
     demo.py
     pipeline_sdxl_instantid_fouriscale.py
   ```

## Code

you can modify the content in`./demo.py#L50-L69` for your usage.

```python

# args
pretrained_model_name_or_path = 'wangqixun/YamerMIX_v8'
weight_dtype = torch.float16
target_height = 2048
target_width = 2048
# set referring image
face_img = load_image("./InstantID/examples/kaifu_resize.png")
pose_img = load_image("./InstantID/examples/poses/pose.jpg")
# set prompt
prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
neg_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"
# InstanID args
controlnet_conditioning_scale=0.8
ip_adapter_scale=0.8
# FouriScale args
start_step = 12 # 20*(30/50)=12              original start_step in FouriScale config is 20
stop_step=21 # # 35*(30/50)=21               original start_step in FouriScale config is 35
# Generation args
num_inference_steps=30 # lower cost of time. original num_inference_steps in FouriScale config is 50
guidance_scale=5.5
```

# Demo

![image-20240403115817325](https://typorastroage.oss-cn-beijing.aliyuncs.com/img/image-20240403115817325.png)

# Acknowledgements

- Development on [InstantID](https://github.com/InstantID/InstantID) code. Thanks for their great works! 😃
- Thanks [FourisScale](https://github.com/LeonHLJ/FouriScale) outstanding research! 💯

 