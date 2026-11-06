from torch.utils.data import DataLoader, Dataset
from data_loaders.motion_loaders.comp_v6_model_dataset import CompMDMGeneratedDataset
import numpy as np
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens, motions, m_raws, m_scale_betas = [], [], [], []

        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            raw = mm_motion['raw_out']
            scale_betas = mm_motion['scale_betas']

            motions.append(motion[None, :])
            m_raws.append(raw[None, :])
            m_scale_betas.append(scale_betas[None, :])
        
        m_lens = np.array(m_lens, dtype=int)
        motions = np.concatenate(motions, axis=0) # [K, T, D]
        m_raws = np.concatenate(m_raws, axis=0) # [K, 4]
        m_scale_betas = np.concatenate(m_scale_betas, axis=0) # [K, 38]
        
        sort_indx = np.argsort(m_lens)[::-1].copy()
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        m_raws = m_raws[sort_indx]
        m_scale_betas = m_scale_betas[sort_indx]
        
        return motions, m_lens, m_raws, m_scale_betas


def get_mdm_loader(model, diffusion, args, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit):
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    dataset = CompMDMGeneratedDataset(model, diffusion, args, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit)

    mm_dataset = MMGeneratedDataset(opt, dataset)

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader