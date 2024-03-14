import ml_collections
import torch
from path import Path


def get_config():
    """Gets the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    config.log_dir = Path('log_dir')
    # Exp info
    #config.dataset_path = Path("/nas2/lait/5000_Dataset/Image/PubLayNet/publaynet")
    config.train_json = "data/cgl_dataset/for_posternuwa/html_format_img_instruct_all/train_llama_numerical.jsonl"
    config.val_json = "data/cgl_dataset/for_posternuwa/html_format_img_instruct_all/val_llama_numerical.jsonl"
    config.train_img_path ="data/cgl_dataset/cgl_inpainting_all"
    config.val_img_path = "data/cgl_dataset/cgl_inpainting_all"

    config.resume_from_checkpoint = None
    config.aug = False

    config.type = "stage2"
    config.vit_model_name = "eva_clip_g"
    config.max_num_comp = 9

    # Training info
    config.seed = 42
    # data specific

    # model specific
    config.lora_r = 64
    config.lora_alpha = 16
    config.lora_dropout = 0.05
    config.prompt_path="src/prompts/algnment.txt"
    config.prompt_template= '[INST] {} [/INST] '
    config.max_txt_len = 320
    config.llama_model = "models/Llama-2-7b-chat-hf"
    config.stage1_model = "log_dir/stage1_all/checkpoints/checkpoint-18/pytorch_model.bin"

    # Training info
    config.log_interval = 1000
    config.sample_interval = 10
    config.save_interval = 4

    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.num_gpus = torch.cuda.device_count()

    config.optimizer.mixed_precision = 'fp16'
    config.optimizer.gradient_accumulation_steps = 1
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.999
    config.optimizer.epsilon = 1e-8
    config.optimizer.weight_decay = 1e-6

    config.optimizer.lr_scheduler = 'cosine'
    config.optimizer.num_warmup_steps = 10000
    config.optimizer.lr = 0.0001

    config.optimizer.num_epochs = 100 #
    config.optimizer.batch_size = 2 #
    config.optimizer.split_batches = False
    config.optimizer.num_workers = 8

    config.optimizer.lmb = 5
    

    if config.optimizer.num_gpus == 0:
        config.device = 'cpu'
    else:
        config.device = 'cuda'
    return config