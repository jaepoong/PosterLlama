import json
from PIL import Image
import easyocr
import numpy as np
import os
from tqdm import tqdm
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to("cuda")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

reader = easyocr.Reader(['ch_sim','en'],gpu=True)


json_data = json.load(open('data/cgl_dataset/for_posternuwa/raw/train.json'))

img_dir_path = "data/cgl_dataset/layout_imgs_6w"
save_path = "data/cgl_dataset/layout_train_6w_translated.json"
result = json_data
i=0
pbar = tqdm(range(len(json_data['annotations'])),desc='desc')
for i,annotations in enumerate(json_data['annotations']):
    path = json_data['images'][i]['file_name']
    img_path=os.path.join(img_dir_path,path[0])
    img = Image.open(img_path)
    te = []
    for j,annot in enumerate(annotations):
        if annot['category_id']==2:
            bbox = annot['bbox']
            croped_image = img.crop([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
            ocr_result = reader.readtext(np.array(croped_image))
            print(ocr_result)
            ocrs = [ocr[1].replace("&nbsp;"," ") for ocr in ocr_result]
            ocrs = " ".join(ocrs)
            
            inputs = tokenizer(ocrs,return_tensors="pt",truncation=True).to("cuda")
            translation_result = tokenizer.batch_decode(model.generate(**inputs,repetition_penalty=1.5,no_repeat_ngram_size=3,max_new_tokens = 30),skip_special_tokens=True)
            
            print(translation_result)
            
            te.append(translation_result)
            result['annotations'][i][j]['text'] = translation_result[0]
            
        else:
            result['annotations'][i][j]['text'] = False
    if i%1000==0:
        print("------"+str(i)+'th result---- ')
        for keys in json_data['annotations'][i]:
            print(keys)
                
    if i%10000==0:
        print("----save checkpoint---")
        with open(save_path,'w') as outfile:
            json.dump(result,outfile)
    pbar.update(1)        

print("----save checkpoint---")
with open(save_path,'w') as outfile:
    json.dump(result,outfile)

pbar.close()