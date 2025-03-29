import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.get_data import get_dataset_loader
from itertools import cycle
from torch import nn
import data_loaders.humanml.utils.paramUtil as paramUtil
from diffusion.nn import mean_flat, sum_flat

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_array
import imageio
from utils.process_smpl_from_hybrik import amass_to_pose, pos2hmlrep
from data_loaders.tensors import collate


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
loss_L1 = nn.L1Loss()


class TrainInpaintingLoop:
    def __init__(self, args, train_platform, model, data, diffusion=None, style_data=None):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.data = data
        self.style_data = style_data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        
        self.style_finetune = args.style_finetune if hasattr(args, "style_finetune") else 0
        self.semantic_guidance = args.semantic_guidance if hasattr(args, "style_finetune") else 0 

        self.skip_steps = args.skip_steps if hasattr(args, "skip_steps") else 0
        if hasattr(args, "skip_steps"):
            self.style_example = True
        else:
            self.style_example = False

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self._load_and_sync_parameters()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.overwrite = args.overwrite

        self.l2_loss = lambda a, b: (a - b) ** 2 
        
        self.diffusion = diffusion
        if diffusion is not None:
            self.schedule_sampler_type = 'uniform'
            self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        
    def _load_and_sync_parameters(self):
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ), strict=False
            )
            assert len(unexpected_keys) == 0
            assert all([k.startswith("motion_enc.") for k in missing_keys])
    

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'opt') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def run_loop(self):
        if self.style_finetune:
            iter_styledata = iter(self.style_data)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
        
            if self.style_finetune:
                try:
                    content_motion, cond_style = next(iter_styledata)
                except:
                    iter_styledata = iter(self.style_data)
                    content_motion, cond_style = next(iter_styledata)
                cond_style['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond_style['y'].items()}
            else:
                content_motion = None
                cond_style = None
            
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                
                self.run_step(motion, cond, content_motion, cond_style)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')
                        # for name, param in self.model.named_parameters():
                        #     if param.requires_grad and name == 'control_seqTransEncoder.layers.1.norm1.weight':
                        #         print(name, param.grad)
                        # print("******************")


                if self.step % self.save_interval == 0:
                    self.save()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, style_batch=None, style_cond=None):
        self.forward_backward(batch, cond, style_batch, style_cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    
    def calculate_bone(self, motion):
        ### motion is normalized motion B J 1 T
        joint_num = 22 if self.args.dataset == 'humanml' else 23
        denorm_motion = self.t2m_data.dataset.t2m_dataset.inv_transform_tensor(motion.permute(0, 2, 3, 1))
        # B 1 T J
        vel = denorm_motion[..., :4]  
        ric_data = denorm_motion[..., 4 : 4 + (joint_num - 1) * 3] 
        ric_data = ric_data.reshape(ric_data.shape[:-1] + ((joint_num - 1), 3)) # x,z are relative to root joint, all face z+
        root_ric = torch.zeros_like(ric_data[..., 0:1, :]).to(ric_data.device)
        root_ric[...,0, 1] = vel[..., 3] 
        ric_data = torch.cat((root_ric, ric_data), dim=-2)

        chains = paramUtil.t2m_kinematic_chain
        bones = []
        for chain in chains:
            for i in range(1, len(chain)):
                bones.append((ric_data[..., chain[i], :] - ric_data[..., chain[i-1], :]).unsqueeze(-2))
        bones = torch.cat(bones, -2)
        return bones              # B 1 T J-1 3

    def forward_backward(self, batch, cond, style_batch, style_cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            micro_content = style_batch
            micro_style_cond = style_cond

            # last_batch = (i + self.microbatch) >= batch.shape[0]
            
            assert self.diffusion is not None

            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            if self.style_finetune:
                if self.args.use_ddim:
                    t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), range(int((self.args.diffusion_steps-self.args.skip_steps)/self.args.diffusion_steps * 20)))
                else:
                    t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), range(self.args.diffusion_steps-self.args.skip_steps))
            else:
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            if self.style_finetune: 
                compute_losses = functools.partial(
                    self.diffusion.few_shot_style_finetune_losses,
                    self.model,
                    micro,
                    t,
                    micro_content,
                    micro_style_cond["y"]["inpainted_motion"],
                    skip_steps=self.args.skip_steps,
                    model_kwargs = micro_style_cond,
                    model_t2m_kwargs = micro_cond,
                    semantic_guidance = self.semantic_guidance,
                    use_ddim=self.args.use_ddim,
                    Ls = self.args.Ls
                    
                )
    

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            if style_batch is None:
                loss = (losses["loss"] * weights).mean()
            else:
                loss = losses["loss"]
            
            if style_batch is None:
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                    )
            else:
                log_loss_dict_style(
                    self.diffusion, t, losses
                )

            self.mp_trainer.backward(loss)
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            # print("******************")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            if self.dataset != 'humanml':
                motion_enc_weights = [e for e in state_dict.keys() if e.startswith('motion_enc.')]
                clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]

                for e in motion_enc_weights:
                    del state_dict[e]

                for e in clip_weights:
                    del state_dict[e]

            else:
                controlmdm_weights = [e for e in state_dict.keys() if e.startswith('controlmdm.')]
                clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
                # mdm_weights = [e for e in state_dict.keys() if e.startswith('mdm_model.')]
                for e in controlmdm_weights:
                    del state_dict[e]

                for e in clip_weights:
                    del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0



def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint(save_dir, mode='model'):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    files = [file for file in os.listdir(save_dir) if (file.endswith('.pt') and file.startswith(mode))]
    steps = [int(file[len(mode):len(mode)+9]) for file in files]
    max_step = sorted(steps)[-1]
    latest_model = f"{mode}{max_step:09d}.pt"
    
    return os.path.join(save_dir, latest_model)


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def log_loss_dict_style(diffusion, ts, losses):
    for key, values in losses.items():
        if key != 'loss':
            logger.logkv_mean(key, values.mean().item())
        else:
            logger.logkv_mean(key, values.item())



def log_motion_encoder_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.item())

def log_motion_encoder_finetune_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())