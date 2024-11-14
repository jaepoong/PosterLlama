import sys
import os
import fire
import torch
import transformers
import json
import copy

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from src.processor.blip_processors import Blip2ImageTrainProcessor,Blip2ImageEvalProcessor,DinoImageProcessor

from src.model.minigpt4 import MiniGPT4
from tqdm import *
import random
from generation import *
from PIL import Image
from html_to_ui import get_bbox,draw_bbox

def split_string_by_delimiter(input_string, delimiter):
    # 입력 문자열을 특정 문자(delimiter)를 기준으로 분할하여 전후 문자열을 반환합니다.
    parts = input_string.split(delimiter)

    # 분할된 문자열이 두 개 이상인 경우, 전후 문자열을 반환합니다.
    if len(parts) >= 2:
        before_delimiter = parts[0]
        after_delimiter = delimiter.join(parts[1:])
        return before_delimiter, after_delimiter
    else:
        # 특정 문자(delimiter)를 찾을 수 없는 경우, 원래 문자열과 빈 문자열을 반환합니다.
        return input_string, ""

# TODO customizing
def main(
    file_path: str = "data/cgl_dataset/for_posternuwa/html_format_img_instruct_mask_all_condition/test_numerical.jsonl",
    base_model: str = "log_dir/train_stage2_with_augment_dino_codellama/checkpoints/checkpoint-16/pytorch_model.bin",
    img_dir: str = "data/cgl_dataset/cgl_inpainting_all",
    device: int=1,
    output_dir: str="log_dir/train_stage2_with_augment_dino_codellama",
    max_new_tokens: int=1024,
    dino=True,
    code_llama = True,
    vis : bool=True,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )
    
    assert file_path, (
        "Please specify a --file_path, e.g. --file_path='/path/to/json_file'"
    )
    assert img_dir, (
        "Please specify a --img_dir, e.g. --file_path='/path/to/img_dir'"
    )
    
    assert device is not None, (
        "Please specify a --device, e.g. --device='0'"
    )
    
    assert output_dir, (
        "Please specify a --output_dir, e.g. --output_dir='/path/to/output_dir'".py
    )
    
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    if dino:
        vit_model_name = "dino_v2"
    else:
        vit_model_name = "eval_clip_g"
    
    if code_llama:
        llama_model = "models/codeLlama-7b-hf"
    else:
        llama_model = "models/Llama-2-7b-chat-hf"
        
    if dino:
        image_processor = DinoImageProcessor()
    else:
        image_processor = Blip2ImageEvalProcessor()
    model = MiniGPT4(lora_r=64,low_resource=False,vit_model = vit_model_name,llama_model = llama_model)
    model.load_state_dict(torch.load(base_model,map_location="cpu"))
    model = model.to(device)
    model.device = device

    model.eval()

    def generate(
        image,
        html_input,
        temperature=0.6,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        **kwargs,
    ):

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                generation_output = model.generate(image,html_input,max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,do_sample=do_sample)
        return generation_output


    with open(file_path, "r") as f:
        content = [json.loads(line) for line in f]
        
    output_file = os.path.join(output_dir, "generated_sample")
    sample_file = os.path.join(output_dir, "generated_sample","samples")
    os.makedirs(output_file,exist_ok=True)
    os.makedirs(sample_file,exist_ok=True)


    res = []
    boxx = []
    boxx_gt = []
    clx = []
    clx_gt = []
    imgs = []
    with tqdm(total=len(content)) as pbar:
        for i,samples in enumerate(content):
            try:
                img_path = os.path.join(img_dir,samples['name'][0][:-4]+".png")
                image = Image.open(img_path)
            except:
                img_path = os.path.join(img_dir,samples['name'][0][0][:-4]+".png")
                image = Image.open(img_path)  
            
            sample_image = copy.deepcopy(image)
            
            size = image.size
            cond_cate_to_size_pos = samples.get("cond_cate_size_to_pos_input_seqs")
            cond_cate_size_to_pos = samples.get("cond_cate_to_size_pos_input_seqs")
            cond_recover_mask_input = samples.get("cond_recover_mask_input_seqs")
            cond_cate_pos_to_size = samples.get("cond_cate_pos_to_size_input_seqs")
            unconditional = samples.get("unconditional_input_seqs")
            
            instruct = []
            instruct_1,answer_3 = split_string_by_delimiter(cond_cate_to_size_pos[0],"<MID>")
            instruct.append(instruct_1)
            instruct_2,answer_3 = split_string_by_delimiter(cond_cate_size_to_pos[0],"<MID>")
            instruct.append(instruct_2)
            instruct_3,answer_3 = split_string_by_delimiter(cond_recover_mask_input[0],"<MID>")
            instruct.append(instruct_3)
            instruct_4,answer_4 = split_string_by_delimiter(cond_cate_pos_to_size[0],"<MID>")
            instruct.append(instruct_4)
            instruct_5,answer_5 = split_string_by_delimiter(unconditional[0],"<MID>")
            instruct.append(instruct_5)

            id_ = samples.get("id_")
            img = image_processor(image)
            img_l = torch.stack([img]*len(instruct))

            generated_sample = generate(img_l,instruct, max_new_tokens=max_new_tokens)
            #cond_cate_size_to_pos = generate(img,cond_cate_size_to_pos, max_new_tokens=max_new_tokens)
            #cond_recover_mask = generate(img,cond_recover_mask_input, max_new_tokens=max_new_tokens)
            
            res.append({
                "cond_cate_to_size_pos": generated_sample[0],
                "cond_cate_size_to_pos": generated_sample[1],
                "cond_recover_mask": generated_sample[2],
                "cond_cate_pos_to_size": generated_sample[3],
                "unconditional" : generated_sample[4],
                "gt_bbox" : samples.get("raw_data"),
                "id": id_,
                "image" : img_path,
                "size" : size
            })
            pbar.update(1)
            
            bbox,label = get_bbox(generated_sample[4])
            img = img_path.split('/')[-1]
            if vis:
                saving_image = draw_bbox(sample_image,bbox,label,verbose=False)
                saving_image.save(os.path.join(sample_file,img))
                
            bbox = [[bb[0],bb[1],bb[0]+bb[2],bb[1]+bb[3]] for bb in bbox] # training 좌표계는 [xl,yl,w,h] -> eval.py는 [xl,yl,xr,yr]
            
            boxx.append(bbox)
            clx.append(label)
            imgs.append(img)
            
            gt = samples.get("raw_data")
            
            gt_bbox,gt_label = gt["bbox"],gt["category"]# gt좌표계는 [x_c,y_c,w,h]
            gt_bbox = [[bb[0]-bb[2]/2,bb[1]-bb[3]/2,bb[0]+bb[2]/2,bb[1]+bb[3]/2] for bb in gt_bbox] 
            boxx_gt.append(gt_bbox)
            clx_gt.append(gt_label)
    
    # save sample
    with open(os.path.join(output_file,file_path.split("/")[-1]), "w") as f:
        for line in res:
            f.write(json.dumps(line) + "\n")
    # save box
    with open(os.path.join(output_file,"box.json"), "w") as f:
        json.dump(boxx,f,indent=2)
    # save box_gt
    with open(os.path.join(output_file,"box_gt.json"), "w") as f:
        json.dump(boxx_gt,f,indent=2)
        
    with open(os.path.join(output_file,"clses.json"),"w") as f:
        json.dump(clx,f,indent=2)

    with open(os.path.join(output_file,"clses_gt.json"),"w") as f:
        json.dump(clx_gt,f,indent=2)

    with open(os.path.join(output_file,"text_order.json"),"w") as f:
        json.dump(imgs,f,indent=2)
    
    print("save done !!")
            
        
if __name__ == "__main__":
    fire.Fire(main)