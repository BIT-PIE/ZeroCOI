import torch
from data_loaders.networks.modules import *
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import dist_utils
from utils.decouple_utils import *
from sample.condition import load_condition


class CompMDMGeneratedDataset(Dataset):

    def __init__(self, model, diffusion, args, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME
        clip_denoised = False  # FIXME
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()
        
        self.cond_mode = args.eval_mode
        self.inv_transform_th = self.dataset.inner_dataset.inv_transform_th
        text_guide_scale = args.guidance_param
        obj_guide_scale = args.guidance_param_noObj

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
                for k, v in model_kwargs['y'].items():
                    if torch.is_tensor(v):
                        model_kwargs['y'][k] = v.to(dist_utils.dev())

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # sample obj trans and orient
                if 'obj' in self.cond_mode:
                    trans, orient = place_obj_global(motion.shape[0], model_kwargs['y']['obj_center_height'], model_kwargs['y']['obj_scales'])
                    model_kwargs['y']['obj_transform_trans'] = trans
                    model_kwargs['y']['obj_transform_rotation'] = orient

                # add CFG scale to batch
                if text_guide_scale != 1.:
                    model_kwargs['y']['text_guide_scale'] = torch.ones(motion.shape[0], device=dist_utils.dev()) * text_guide_scale
                if obj_guide_scale != 1. and 'obj' in self.cond_mode:
                    model_kwargs['y']['obj_guide_scale'] = torch.ones(motion.shape[0], device=dist_utils.dev()) * obj_guide_scale

                cond_fn = load_condition(args, self.inv_transform_th) if 'obj' in self.cond_mode else None
                
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    sample, raw_out = sample_fn(
                        model,
                        motion.shape,
                        noise=None,
                        clip_denoised=clip_denoised,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu(),
                                    'raw_out': raw_out[bs_i].cpu(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'scale_betas': model_kwargs['y']['scale_betas'][bs_i].cpu(),
                                    'species': model_kwargs['y']['species'][bs_i],
                                    'obj_id': model_kwargs['y']['obj_id'][bs_i] if 'obj_id' in model_kwargs['y'] else None,
                                    'obj_transform_trans': model_kwargs['y']['obj_transform_trans'][bs_i].cpu() if 'obj_transform_trans' in model_kwargs['y'] else None,
                                    'obj_transform_rotation': model_kwargs['y']['obj_transform_rotation'][bs_i].cpu() if 'obj_transform_rotation' in model_kwargs['y'] else None,
                                    'obj_scales': model_kwargs['y']['obj_scales'][bs_i].cpu() if 'obj_scales' in model_kwargs['y'] else None,
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu(),
                                        'raw_out': raw_out[bs_i].cpu(),
                                        'scale_betas': model_kwargs['y']['scale_betas'][bs_i].cpu(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'species': model_kwargs['y']['species'][bs_i],
                                    'obj_id': model_kwargs['y']['obj_id'][bs_i] if 'obj_id' in model_kwargs['y'] else None,
                                    'obj_transform_trans': model_kwargs['y']['obj_transform_trans'][bs_i].cpu() if 'obj_transform_trans' in model_kwargs['y'] else None,
                                    'obj_transform_rotation': model_kwargs['y']['obj_transform_rotation'][bs_i].cpu() if 'obj_transform_rotation' in model_kwargs['y'] else None,
                                    'obj_scales': model_kwargs['y']['obj_scales'][bs_i].cpu() if 'obj_scales' in model_kwargs['y'] else None,
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len, scale_betas, species = data['cap_len'], data['scale_betas'], data['species']

        obj_id, obj_transform_trans, obj_transform_rotation, obj_scales, raw_out = \
            data['obj_id'], data['obj_transform_trans'], data['obj_transform_rotation'], data['obj_scales'], data['raw_out']
        
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = torch.tensor(np.concatenate(pos_one_hots, axis=0))
        word_embeddings = torch.tensor(np.concatenate(word_embeddings, axis=0))

        return word_embeddings, pos_one_hots, caption, sent_len, motion, scale_betas, m_length, raw_out,  \
            obj_id, obj_transform_trans, obj_transform_rotation, obj_scales, species
