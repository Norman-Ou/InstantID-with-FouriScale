import torch
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers.models import ControlNetModel
import cv2
import numpy as np
from compel import Compel, ReturnedEmbeddingsType
import math
import time
from PIL import Image

from pipeline_sdxl_instantid_fouriscale import StableDiffusionXLInstantIDFouriScalePipeline

from fouriscale.models import TrainingFreeAttnProcessor
from fouriscale.utils import read_base_settings, read_layer_settings, find_smallest_padding_pair
from fouriscale.aux_xl import list_layers

from InstantID.pipeline_stable_diffusion_xl_instantid import draw_kps
from diffusers.utils import load_image
from insightface.app import FaceAnalysis

def resize_and_pad(image_pil, size):
    original_size = image_pil.size
    target_w, target_h = size
    
    aspect_ratio = original_size[0] / original_size[1]
    if (target_w / target_h) > aspect_ratio:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
    else:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    
    resized_image = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    new_image = Image.new("RGB", (target_w, target_h))
    
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    
    new_image.paste(resized_image, (left, top))
    
    return new_image


def main():
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

    # Load Fouriscale Setting
    layer_settings = read_layer_settings("./fouriscale/assets/layer_settings/sdxl.txt")
    base_settings = read_base_settings("./fouriscale/assets/base_settings/sdxl.txt")

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=weight_dtype
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer_2", torch_dtype=weight_dtype
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype
    )
    unet.set_attn_processor({name: TrainingFreeAttnProcessor(name) for name in list_layers})

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    # # prepare InstantID model under ./InstantID/checkpointsInstantID
    face_adapter = './InstantID/checkpoints/ip-adapter.bin'
    controlnet_path = './InstantID/checkpoints/ControlNetModel'

    # prepare 'antelopev2' under ./InstantID/models
    app = FaceAnalysis(name='antelopev2', root='./InstantID/', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipeline = StableDiffusionXLInstantIDFouriScalePipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=noise_scheduler,
        controlnet=controlnet,
    )
    pipeline.cuda()
    unet.eval()

    # load adapter
    pipeline.load_ip_adapter_instantid(face_adapter)

    # init compel for longer prompt
    compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,requires_pooled=[False, True])

    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_emb = face_info['embedding']

    # prepare referring pose
    pose_info = app.get(cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR))
    pose_info = sorted(pose_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_kps = draw_kps(image_pil=pose_img, kps=pose_info['kps'])
    # resize face key points image size to target size to gain a better result!
    face_kps = resize_and_pad(face_kps, size=(target_width, target_height))

    # encode prompt and negative prompt
    p_prompt_embeds, p_prompt_pooled = compel(prompt)
    n_prompt_embeds, n_prompt_pooled = compel(neg_prompt)

    pipeline.enable_vae_tiling()

    # FouriScale
    base_size, aspect_ratio = find_smallest_padding_pair(target_height, target_width, base_settings)
    print(f"Using reference size {base_size}")
    dilation = max(math.ceil(target_height / base_size[0]),
                   math.ceil(target_width / base_size[1]))

    start_time = time.time()
    image = pipeline(
        prompt_embeds=p_prompt_embeds,
        pooled_prompt_embeds=p_prompt_pooled,
        negative_prompt_embeds=n_prompt_embeds,
        negative_pooled_prompt_embeds=n_prompt_pooled,
        image_embeds=face_emb,
        image=face_kps,
        width=target_width,
        height=target_height,
        original_size=base_size,
        target_size=base_size,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        ip_adapter_scale=ip_adapter_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        dilation=dilation,
        start_step=start_step,
        stop_step=stop_step,
        layer_settings=layer_settings,
        base_size=base_size,
        progressive=True,
    ).images[0]
    end_time = time.time()
    print(f"Time: {end_time - start_time}") 
    image.save(f'instantid_fouriscale_{target_width}_{target_height}.jpg')

if __name__ == "__main__":
    main()














