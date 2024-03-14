import os
import torch
from accelerate import Accelerator
from ml_collections import config_flags,config_dict
from src.common.utils import set_seed
from absl import flags, app
from src.common.logger_set import LOG

from src.dataset.custom_dataset import RawFileDataset
from src.dataset.caption_dataset import ChainDataset
from src.model.minigpt4 import MiniGPT4
#from src.trainer.trainer import Trainer
from loguru import logger
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.envirion["DS_SKIP_CUDA_CHECK"] = "1"
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", "Training configuration.",
                                lock_config=False)
flags.DEFINE_string("workdir", default='test', help="Work unit directory.")
flags.mark_flags_as_required(["config",])

def main(*args, **kwargs):
    config = init_job()
    log_file_path = os.path.join(config.log_dir,"training_log.txt")
    logger.add(log_file_path, rotation="10 MB")

    accelerator = Accelerator(
        split_batches=config.optimizer.split_batches,
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps,
        mixed_precision=config.optimizer.mixed_precision,
        project_dir=config.log_dir,
        dispatch_batches = False
    )
    
    if accelerator.is_main_process:
        logger.info("Loading data.")

    if config.type == "stage1":
        train_data = ChainDataset(config.train_img_path,config.vit_model_name)
        val_data = ChainDataset(config.train_img_path,config.vit_model_name)
    elif config.type == "stage2":
        train_data = RawFileDataset(config.train_json, img_file_path = config.train_img_path, vit_model_name = config.vit_model_name, aug = config.aug)
        val_data = RawFileDataset(config.val_json, img_file_path = config.val_img_path, vit_model_name = config.vit_model_name, aug = config.aug)
        logger.info(f"Train data length : {len(train_data)} \n Val_data_length : {len(val_data)} ")
    else:
        ValueError("Invalid value for 'config.type', you must set it for stage1 or stage2")

    if accelerator.is_main_process:
        logger.info(accelerator.state)
        logger.info("............Creating model...........")
        
    if accelerator.is_main_process:
        logger.info("----"*10+"config"+"---"*10+"\n")
        logger.info(config)
    
    model = MiniGPT4(
        vit_model = config.vit_model_name,
        lora_r = config.lora_r,
        prompt_path = config.prompt_path,
        prompt_template = config.prompt_template,
        llama_model = config.llama_model,
        max_txt_len = config.max_txt_len
    )

    if config.type =="stage2" and config.stage1_model:
        print("----------Load Pretrained Stage1 Model-------------")
        state = torch.load(config.stage1_model,map_location="cpu")
        model.load_state_dict(state,strict=False)
        name = ["llama_proj.weight","llama_proj.bias"]
        for n,p in model.named_parameters():
            if n in name:
                p.requires_grad = False
        del state
    
    if config.type =="stage2":
        from src.trainer.trainer_2 import Trainer
        trainer = Trainer(
            accelerator,
            model,
            train_data,
            val_data,
            config,
            logger,
            resume_from_checkpoint=config.resume_from_checkpoint
        )
    else:
        from src.trainer.trainer_1 import Trainer
        trainer = Trainer(
            accelerator,
            model,
            train_data,
            val_data,
            config,
            logger,
            resume_from_checkpoint=config.resume_from_checkpoint
        )
    trainer.train()




def init_job():
    config = FLAGS.config
    config.log_dir = config.log_dir / FLAGS.workdir
    config.ckpt_dir = config.log_dir / 'checkpoints'
    config.samples_dir = config.log_dir / 'samples'
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)
    os.makedirs(config.ckpt_dir, exist_ok=True)
    set_seed(config.seed)
    return config

if __name__=='__main__':
    app.run(main)