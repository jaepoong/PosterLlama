import math
import os
import random
import shutil

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from src.common.logger_set import LOG
import time
class Trainer:
    def __init__(self, accelerator: Accelerator, model, train_data, val_data, conf,logger,
        resume_from_checkpoint: str = None):
        self.accelerator = accelerator
        self.model = model
        self.model.device = self.accelerator.device
        self.device = self.accelerator.device
        self.train_data = train_data
        self.val_data = val_data
        self.conf = conf
        self.logger = logger
        print(self.device)

        
        # training concept
        self.num_epochs = conf.optimizer.num_epochs
        self.save_interval = conf.save_interval

        # optimizer concept
        self.lr = conf.optimizer.lr
        self.weight_decay = conf.optimizer.weight_decay
        self.beta1 = conf.optimizer.beta1
        self.beta2 = conf.optimizer.beta2

        # scheduler concept
        print("-------------Init Optimizer----------")
        self.scheduler = conf.optimizer.lr_scheduler
        self.num_warmup_steps = conf.optimizer.num_warmup_steps
        self.gradient_accumulation_steps = conf.optimizer.gradient_accumulation_steps
        self.batch_size = conf.optimizer.batch_size
        self.num_workers = conf.optimizer.num_workers
        #for n, p in self.model.named_parameters():
        #    if p.requires_grad:
        #        print(n)
        self.optimizer = self.init_optimizer(model)

        train_loader = DataLoader(train_data, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers)
        val_loader = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        if conf.optimizer.num_step_per_epoch:
            self.num_step_per_epoch = conf.optimizer.num_step_per_epoch
        else:
            self.num_step_per_epoch = len(train_loader)

        lr_scheduler = get_scheduler(self.scheduler,
                                     self.optimizer,
                                     num_warmup_steps=self.num_warmup_steps * self.gradient_accumulation_steps,
                                     num_training_steps=(self.num_step_per_epoch) * self.num_epochs)

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = accelerator.prepare(
            model, self.optimizer, train_loader, val_loader, lr_scheduler
        )
        if self.accelerator.is_main_process:
            self.logger.info((self.model.device, self.device))

        self.total_batch_size = self.batch_size*accelerator.num_processes*self.gradient_accumulation_steps
        self.num_update_steps_per_epoch = math.ceil(self.num_step_per_epoch)/self.gradient_accumulation_steps
        self.max_train_steps = self.num_epochs*self.num_update_steps_per_epoch
        if accelerator.is_main_process:
            self.logger.info("***** Running training *****")
            self.logger.info(f"  Num examples = {self.num_update_steps_per_epoch}")
            self.logger.info(f"  Num Epochs = {self.num_epochs}")
            self.logger.info(f"  Instantaneous batch size per device = {self.batch_size}")
            self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
            self.logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
            self.logger.info(f"  Total optimization steps = {self.max_train_steps}")

        self.first_epoch = 0
        self.global_step = 0
        self.log_interval = conf.log_interval

        self.resume_from_checkpoint = resume_from_checkpoint
        if resume_from_checkpoint:
            LOG.print(f"Resuming from checkpoint {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            last_epoch = int(resume_from_checkpoint.split("-")[1])
            self.global_step = last_epoch * self.num_update_steps_per_epoch
            self.first_epoch = last_epoch
            self.resume_step = 0
       
        """
        ## 실험용
        save_path = self.conf.ckpt_dir / f"checkpoint-{0}/"
        # delete folder if we have already 5 checkpoints
        self.accelerator.save_state(save_path)
        self.model.module.save_pretrained(save_path)
        LOG.info(f"Saving checkpoint to {save_path}")"""
        
    def init_optimizer(self, model):
        p_wd, p_non_wd = [], []
        num_parameters = 0
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n: 
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        self.logger.info("number of trainable parameters: %d" % num_parameters)
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": self.weight_decay,
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=self.lr,
            weight_decay= self.weight_decay,
            betas=(self.beta1, self.beta2),
        )
        return optimizer
    
    def train(self):
        start_time = time.time()
        best_epoch = 0


        for epoch in range(self.first_epoch, self.num_epochs):
            self.train_epoch(epoch)

        
    def train_epoch(self,epoch):
        self.model.train()
        device = self.model.device
        # pbar setting
        progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        losses = {}

        for step in range(int(self.num_update_steps_per_epoch)):
            sample = next(iter(self.train_dataloader))
            sample['image'] = sample['image'].to(device)
            # train with acclerator, if want efficient high batch training, set high accumulation values
            with self.accelerator.accumulate(self.model):
                loss = self.model(sample)["loss"]
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            progress_bar.update(1)
            losses.setdefault("loss",[]).append(loss.detach().item())
            if self.accelerator.is_main_process & self.accelerator.sync_gradients:
                self.global_step += 1
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0],
                        "step": self.global_step}
                progress_bar.set_postfix(**logs)
        progress_bar.close()

        save_path = self.conf.ckpt_dir / f"checkpoint-{epoch}/"
        # delete folder if we have already 5 checkpoints
        if self.conf.ckpt_dir.exists() & self.accelerator.is_main_process:
            ckpts = list(self.conf.ckpt_dir.glob("checkpoint-*"))
            # sort by epoch
            ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))
            #if len(ckpts) > 5:
            #    self.logger.info(f"Deleting checkpoint {ckpts[0]}")
            #    shutil.rmtree(ckpts[0])

        if epoch%self.save_interval==0 or epoch==self.num_epochs-1 & self.accelerator.is_main_process:
            #self.accelerator.save_state(save_path)
            self.accelerator.save_model(self.model,save_path,"50GB",safe_serialization=False)
            #self.model.save_pretrained(save_path)
            self.logger.info(f"Saving checkpoint to {save_path}")

        if self.conf.type=="stage1":
            if self.accelerator.is_main_process:
                with torch.no_grad():
                    self.model.eval()
                    samp = next(iter(self.val_dataloader))
                    out = self.model.module.generate(samp['image'].to(device),max_new_tokens = 320)
                    self.logger.info(f"Epoch {epoch}, Step {step}, \n sample : {out}")
                    self.logger.info(logs)
                    del samp
                self.model.train()
        else:
            if self.accelerator.is_main_process:
                with torch.no_grad():
                    self.model.eval()
                    samp = next(iter(self.val_dataloader))
                    out = self.model.module.generate(samp['image'].to(device),samp['input'],max_new_tokens = 320)
                    self.logger.info(f"Epoch {epoch}, Step {step}, \n sample : {out}")
                    self.logger.info(logs)
                    del samp
                self.model.train()
            
        self.accelerator.wait_for_everyone()

#[6] NVIDIA A100 80GB PCIe | 57°C,  99 % | 65276 / 81920 MB | poong(65274M)  32 batch 4
#[7] NVIDIA A100 80GB PCIe | 41°C, 100 % | 65448 / 81920 MB | poong(65446M)


#[6] NVIDIA A100 80GB PCIe | 56°C,   0 % | 66440 / 81920 MB | poong(66438M)  64 batch 4
#[7] NVIDIA A100 80GB PCIe | 40°C, 100 % | 66204 / 81920 MB | poong(66202M)

#[6] NVIDIA A100 80GB PCIe | 54°C,  99 % | 69424 / 81920 MB | poong(69422M) 128 batch 4
#[7] NVIDIA A100 80GB PCIe | 43°C, 100 % | 69468 / 81920 MB | poong(69466M)
