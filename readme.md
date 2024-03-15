# PosterLlama

# Unconditional Generated Output
<img src = "asset/generated_samples.png" width="50%" height="50%">


# Setup
```bash
conda create -n PosterLlama python==3.9
conda activate PosterLlama
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install requirments.txt
```

### Model Preparation
We utilize [LLaMA2-7B-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [CodeLLaMA-7B](https://huggingface.co/codellama/CodeLlama-7b-hf) as our backbone.
You can download the models and place them under the ``./models`` directory.

# Training
Basic setting is about dino+code_llama model.
For diverse training, you can choose more setup at ``./src/common/configs*.py``

### First-Stage Training
```bash

 DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --gpu_ids='all'  main.py  --config src/common/configs_stage2_dino_code_llama.py --workdir codellama

```

### Second-Stage Training
For second stage training, we utilize deepspeed stage-2. So before training, we recommend to setup the accelerate config.
```bash

 DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --gpu_ids='all'  main.py  --config src/common/configs_stage2_dino_code_llama.py --workdir codellama

```