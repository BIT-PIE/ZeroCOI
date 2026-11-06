# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import functools
import os
import copy
import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_utils
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from utils.decouple_utils import *
from utils.model_utils import load_model_wo_shape

class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.model_avg = None
        if self.args.use_ema:
            self.model_avg = copy.deepcopy(self.model)
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
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

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        # setup optimizer with different lr for RTDecoder
        param_to_name = {p: name for name, p in self.model.named_parameters()}
        
        rt_decoder_params = []
        other_params = []
        
        for p in self.mp_trainer.master_params:
            name = param_to_name.get(p, "")
            if name.startswith("c_RT_decoder."):
                rt_decoder_params.append(p)
            else:
                other_params.append(p)
        
        param_groups = [
            {'params': other_params, 'lr': self.lr, 'initial_lr': self.lr},
            {'params': rt_decoder_params, 'lr': self.lr * 2.0, 'initial_lr': self.lr * 2.0},
        ]

        if self.args.use_ema:
            self.opt = AdamW(
                param_groups,
                weight_decay=self.weight_decay,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt = AdamW(
                param_groups,
                weight_decay=self.weight_decay
            )
        
        # Model was resumed, either due to a restart or a checkpoint
        # being specified at the command line.
        # if self.resume_step:
        #     self._load_optimizer_state()

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_utils.dev() != 'cpu':
            self.device = torch.device(dist_utils.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_utils.load_state_dict(
                        resume_checkpoint, map_location=dist_utils.dev()
                    )
            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict['model_avg']
                if self.cond_mode == 'text':
                    load_model_wo_shape(self.model, state_dict)
                    load_model_wo_shape(self.model_avg, state_dict_avg)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model_avg.load_state_dict(state_dict_avg, strict=False)
            else:
                if self.cond_mode == 'text':
                    load_model_wo_shape(self.model, state_dict)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
                if self.args.use_ema:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=False)

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_utils.load_state_dict(
                opt_checkpoint, map_location=dist_utils.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            tqdm.write(f'Starting epoch {epoch}')
            progress = tqdm(self.data, position=0, dynamic_ncols=True)
            metric_line = tqdm(total=1, position=1, leave=True, bar_format="{desc}", dynamic_ncols=True)
            metric_line.set_description_str("losses: waiting...")
            for motion, cond in progress:
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                step_metrics = self.run_step(motion, cond)
                metric_text = ' | '.join([f"{k}:{v:.4f}" for k, v in step_metrics.items()]) if step_metrics else "waiting..."
                metric_line.set_description_str(f"losses: {metric_text}", refresh=True)
                
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    if self.args.use_ema:
                        self.model_avg.eval()
                    self.evaluate()
                    self.model.train()
                    if self.args.use_ema:
                        self.model_avg.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            
            tqdm.write("")
            
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        return

    def run_step(self, batch, cond):
        step_metrics = self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self.update_average_model()
        self._anneal_lr()
        self.log_step()
        return step_metrics
    
    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        step_metrics = None
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]

            # get obj trans and orient
            if 'obj' in self.cond_mode:
                trans, orient = place_obj_global(micro.shape[0], micro_cond['y']['obj_center_height'], micro_cond['y']['obj_scales'])
                micro_cond['y']['obj_transform_trans'] = trans
                micro_cond['y']['obj_transform_rotation'] = orient

            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_utils.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                weights=weights,
                step = self.step,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            step_metrics = self.diffusion.build_step_metrics({k: v * weights for k, v in losses.items()})
            log_loss_dict(
                self.diffusion, self.args, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.mp_trainer.master_params, max_norm=15.0)
        return step_metrics

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        for param_group in self.opt.param_groups:
            # use initial_lr to maintain the lr ratio between param groups
            initial_lr = param_group.get('initial_lr', self.lr)
            param_group["lr"] = initial_lr * (1 - frac_done)
    
    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):

            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
                for e in clip_weights:
                    del state_dict[e]
            
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            del_clip(state_dict)
            
            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            tqdm.write("saving model...")
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


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, args, ts, losses):
    for key, values in losses.items():
        if key == "loss":
            values += losses.get("video_SDS", 0.) * args.lambda_vsd
        logger.logkv_mean(key, values.mean().item())
        
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
