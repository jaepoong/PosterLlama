import torch
import numpy as np
from PIL import Image
import json
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import DPTImageProcessor, DPTForDepthEstimation,DPTFeatureExtractor

import torch
from typing import List
from torchvision import transforms

from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image

from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from dreamsim import dreamsim
import os
import copy
from tqdm import tqdm

def concatenate_images(images: List[Image.Image]) -> Image.Image:
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)

    result_image = Image.new("RGB", (total_width, heights[0]))


    x_offset = 0
    for img in images:
        result_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return result_image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Depth estimation
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

# image caption operator
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

# controllnet operator
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()
controlnet_conditioning_scale = 0.5
# clip operator
to_tensor = transforms.ToTensor()
#metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
model, preprocess = dreamsim(pretrained=True,cache_dir=".cache") 
    
img_path = "data/PKU_PosterLayout/train/inpainted_poster"
img_paths = os.listdir(img_path)
img_paths = img_paths[len(img_paths)//2:] 

consistency_num = 10
aug_num = 3
aug_save_path = "data/PKU_PosterLayout/train/pku_aug"
cont = {}#
os.makedirs(aug_save_path,exist_ok=True)

progress_bar = tqdm(total = len(img_paths))
progress_bar.set_description("augmentation")
for i in range(len(img_paths)):
    samples = []
    aug_score = []
    
    orig_image = Image.open(os.path.join(img_path,img_paths[i]))#
    size = orig_image.size
    orig_image = orig_image.resize((1024,1024))
    # text description of original image
    inputs = blip_processor(images=orig_image, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip_model.generate(**inputs,max_new_tokens=100)
    caption_orig = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    prompt = "please generate" + caption_orig + " in colorful advertisement poster design without text"
    # estimate depth
    depth_image = get_depth_map(orig_image)
    

    # controllnet estimation
    generated_image = pipe(
        [prompt]*consistency_num,
        image=orig_image,
        control_image=depth_image,
        strength=0.99,
        num_inference_steps=10,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images
    
    img1 = preprocess(orig_image)
    for k in range(len(generated_image)):
        img2 = preprocess(generated_image[k])
        distance = model(img1.to(device),img2.to(device))
        samples.append(generated_image[k])
        aug_score.append(distance.item())

    max_indices = sorted(range(len(aug_score)), key=lambda i: aug_score[i], reverse=True)[:aug_num]
    selected_images = [samples[i] for i in max_indices]
    
    orig_image.resize(size).save(os.path.join(aug_save_path,img_paths[i]))
    for j,selected_image in enumerate(selected_images):
        selected_save_path = img_paths[i].split(".")[0]+f"_aug{j}."+img_paths[i].split(".")[1]
        selected_image.resize(size).save(os.path.join(aug_save_path, selected_save_path))
    
    progress_bar.update(1)

progress_bar.close()