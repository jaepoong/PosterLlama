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
from torchmetrics.multimodal.clip_score import CLIPScore

from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image

from torchvision import transforms
#from torchmetrics.multimodal.clip_score import CLIPScore
from dreamsim import dreamsim

import os
import copy
from dataclasses import dataclass
import tyro
from tqdm import tqdm

def concatenate_images(images: List[Image.Image]) -> Image.Image:
    # 이미지 크기 가져오기
    widths, heights = zip(*(img.size for img in images))

    # 총 가로 너비 계산
    total_width = sum(widths)

    # 새로운 이미지 생성
    result_image = Image.new("RGB", (total_width, heights[0]))

    # 이미지 이어붙이기
    x_offset = 0
    for img in images:
        result_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return result_image

def get_depth_map(image,depth_estimator,feature_extractor,device):
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

@dataclass
class Args:
    file_path : str = "data/cgl_dataset/layout_train_6w_fixed_v2.json"
    
    img_path : str = "data/cgl_dataset/cgl_inpainting_all"
    
    consistency_num : int = 10
    
    aug_num : int =3
    
    aug_save_path = "data/cgl_dataset/sam"
    
    quarter : int = 0
    
    device : int = 7


if __name__=="__main__":
    args = tyro.cli(Args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    
    # Depth estimation
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    
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
        torch_dtype=torch.float16,
    ).to(device)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        device = args.device
    )#.to(device)
    pipe.enable_model_cpu_offload()
    controlnet_conditioning_scale = 0.5
    # clip operator
    to_tensor = transforms.ToTensor()
    #metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    model, preprocess = dreamsim(pretrained=True,cache_dir=".cache") 
    
    
    with open(args.file_path, "r") as f:
        #content = [json.loads(line) for line in f]
        content = json.load(f)
    img_path = args.img_path
    #cont = []
    cont = {}#
    cont["categories"] = content["categories"]#
    cont["annotations"] = []
    cont["images"] = []
    os.makedirs(args.aug_save_path,exist_ok=True)
    # augment dataset
    total = len(content["images"])
    #quarter = total//4
    #start = quarter *args.quarter
    #end = min((quarter)*(args.quarter+1), total)
    progress_bar = tqdm(total = total)
    #print(f"start : {start}   end : {end}")
    for i in range(len(content["images"])):
    #for i in range(total):
        samples = []
        aug_score = []
        
        # original image
        orig_image = Image.open(os.path.join(img_path, content["images"][i]["file_name"][:-4]+".png"))
        size = orig_image.size
        orig_image = orig_image.resize((1024,1024))
        
        # text description of original image
        inputs = blip_processor(images=orig_image, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip_model.generate(**inputs)
        caption_orig = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # make generation text
        prompt = "please generate " + caption_orig + " in advertisement poster design without any text description."
        # estimate depth
        depth_image = get_depth_map(orig_image,depth_estimator,feature_extractor,device)

        # controllnet estimation
        generated_image = pipe(
            [prompt]*args.consistency_num,
            image=orig_image,
            control_image=depth_image,
            strength=0.99,
            num_inference_steps=15,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images
        
        img1 = preprocess(orig_image)
        for k in range(len(generated_image)):
            img2 = preprocess(generated_image[k])
            distance = model(img1.to(device),img2.to(device))
            samples.append(generated_image[k])
            aug_score.append(distance.item())
        
        # match generated samples
        max_indices = sorted(range(len(aug_score)), key=lambda i: aug_score[i], reverse=True)[:args.aug_num]
        selected_images = [samples[i] for i in max_indices]
        
        cont["annotations"].append(content["annotations"][i])#.append(sample)
        samp = copy.deepcopy(content["images"][i])
        samp["file_name"] = [samp["file_name"]]
        orig_image.resize(size).save(os.path.join(args.aug_save_path,content["images"][i]["file_name"][:-4]+".png"))
        
        for j,selected_image in enumerate(selected_images):
            selected_save_path = content["images"][i]["file_name"].split(".")[0]+f"_aug{j}."+content["images"][i]["file_name"].split(".")[1]
            selected_image.resize(size).save(os.path.join(args.aug_save_path,selected_save_path))
            #cont["annotations"].append(cont["annotations"][i])
            #samp = copy.deepcopy(content["images"][i])
            samp["file_name"].append(selected_save_path)
            #cont["images"].append(samp)
        cont["images"].append(samp)
        progress_bar.update(1)
    progress_bar.close()
            
    save_path = args.file_path.split(".")[0]+f"_aug."+args.file_path.split(".")[1]
    with open(save_path,"w") as f:
        json.dump(cont,f,indent=2)

    
    
    