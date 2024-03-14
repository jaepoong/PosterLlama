import ml_collections
import torch
from path import Path


def get_config():
    """Gets the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    config.log_dir = Path('log_dir')
    # Exp info
    config.train_img_path = ["/data1/poong/data/cc_sbu_laion/laion_dataset/{00000..10487}.tar",
                             "/data1/poong/data/cc_sbu_laion/cc_sbu_dataset/{00000..01254}.tar"]
    config.val_img_path = ["/data1/poong/data/cc_sbu_laion/laion_dataset/10488.tar",
                           "/data1/poong/data/cc_sbu_laion/cc_sbu_dataset/01255.tar"]
    config.resume_from_checkpoint = None

    config.type = "stage1"
    config.vit_model_name = "eva_clip_g" #dino
    config.max_num_comp = 9

    # Training info
    config.seed = 42
    # data specific

    # model specific
    config.lora_r=0
    config.prompt_path="src/prompts/step1.txt"
    config.prompt_template= '[INST] {} [/INST] '
    config.max_txt_len = 64
    config.llama_model = "models/Llama-2-7b-chat-hf"

    # Training info
    config.log_interval = 10000
    config.sample_interval = 10
    config.save_interval = 2

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
    config.optimizer.num_step_per_epoch = 5000
    config.optimizer.lr = 0.0001

    config.optimizer.num_epochs = 20 #
    config.optimizer.batch_size = 48 #
    config.optimizer.split_batches = False
    config.optimizer.num_workers = 16

    config.optimizer.lmb = 5
    

    if config.optimizer.num_gpus == 0:
        config.device = 'cpu'
    else:
        config.device = 'cuda'
    return config