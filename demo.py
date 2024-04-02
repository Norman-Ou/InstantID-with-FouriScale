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
    
    # 计算缩放比例，保持宽高比不变
    aspect_ratio = original_size[0] / original_size[1]
    if (target_w / target_h) > aspect_ratio:
        # 如果目标宽高比大于原始宽高比，则以高为基准进行缩放
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
    else:
        # 如果目标宽高比小于或等于原始宽高比，则以宽为基准进行缩放
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    
    # 缩放图像
    resized_image = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 创建一个新的空白图像，其尺寸为目标尺寸
    new_image = Image.new("RGB", (target_w, target_h))
    
    # 计算关键点图像在新图像上的位置
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    
    # 将缩放后的图像粘贴到新图像上，使其居中
    new_image.paste(resized_image, (left, top))
    
    return new_image


def main():
    # args
    pretrained_model_name_or_path = 'wangqixun/YamerMIX_v8'
    weight_dtype = torch.float16
    pixel_height = 1534
    pixel_width = 1024

    # Fouriscale Setting
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

    # load referring face image and referring pose image
    face_img = load_image("/home/ruizhe.ou/ComfyUI/input/test1.jpeg")
    pose_img = load_image("/home/ruizhe.ou/ComfyUI/input/test1.jpeg")

    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_emb = face_info['embedding']

    # prepare referring pose
    pose_info = app.get(cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR))
    pose_info = sorted(pose_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_kps = draw_kps(image_pil=pose_img, kps=pose_info['kps'])
    # resize face key points image size to target size
    face_kps = resize_and_pad(face_kps, size=(pixel_width, pixel_height))

    prompt = "isometric style,sensitive,1girl,solo,long_hair,red_hair,navel,cowboy_shot,midriff,pants,black_eyes,lips,crop_top,looking_to_the_side,torn_clothes,makeup,buttons,black_pants,jeans,realistic,torn_pants. vibrant, beautiful, crisp, detailed, ultra detailed, intricate"
    prompt = "isometric style,general,sensitive,1girl,long_hair,looking_at_viewer,brown_hair,shirt,black_hair,brown_eyes,jewelry,jacket,earrings,solo_focus,necklace,blurry,black_eyes,lips,parted_bangs,depth_of_field,blurry_background,denim,jeans,realistic,denim_jacket.vibrant, beautiful, crisp, detailed, ultra detailed, intricate"
    n_prompt = "deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic"
    p_prompt_embeds, p_prompt_pooled = compel(prompt)
    n_prompt_embeds, n_prompt_pooled = compel(n_prompt)

    pipeline.enable_vae_tiling()

    # FouriScale
    base_size, aspect_ratio = find_smallest_padding_pair(pixel_height, pixel_width, base_settings)
    print(f"Using reference size {base_size}")
    dilation = max(math.ceil(pixel_height / base_size[0]),
                   math.ceil(pixel_width / base_size[1]))

    start_time = time.time()
    image = pipeline(
        prompt_embeds=p_prompt_embeds,
        pooled_prompt_embeds=p_prompt_pooled,
        negative_prompt_embeds=n_prompt_embeds,
        negative_pooled_prompt_embeds=n_prompt_pooled,
        image_embeds=face_emb,
        image=face_kps,
        width=pixel_width,
        height=pixel_height,
        original_size=base_size,
        target_size=base_size,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5.5,
        dilation=dilation,
        start_step=12,
        stop_step=21,
        layer_settings=layer_settings,
        base_size=base_size,
        progressive=True,
    ).images[0]
    end_time = time.time()
    print(f"Time: {end_time - start_time}") 
    image.save('result_fouriscal.jpg')

if __name__ == "__main__":
    main()














