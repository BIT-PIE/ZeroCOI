import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from data_loaders.utils.word_vectorizer import WordVectorizer
from data_loaders.utils.get_opt import get_opt


'''For use of training matching model, and evaluations'''
class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = opt.min_motion_length
        if self.opt.fixed_len > 0:
            self.max_length = self.opt.fixed_len
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        name_list = []
        length_list = []

        _split = os.path.basename(split_file).replace('.txt', '')
        cache_path = os.path.join(opt.cache_dir, 'dataset', self.opt.dataset_name + '_' + _split + '.npy')
        if opt.use_cache and os.path.exists(cache_path):
            print(f'Loading motions from cache file [{cache_path}]...')
            _cache = np.load(cache_path, allow_pickle=True)[None][0]
            name_list, length_list, data_dict = _cache['name_list'], _cache['length_list'], _cache['data_dict']
        else:
            for name in tqdm(id_list):
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                text_data = []
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = self.parse_text_line(line)
                        text_data.append(text_dict)

                scale_betas = np.load(pjoin(opt.betas_dir, text_data[0]['species'].replace(' ', '_') + '.npy'))

                data_dict[name] = {'motion': motion, 'scale_betas': scale_betas, 'length': len(motion), 'text': text_data}
                name_list.append(name)
                length_list.append(len(motion))

            name_list, length_list = zip(*sorted(zip(name_list, length_list), key=lambda x: x[1]))
            print(f'Saving motions to cache file [{cache_path}]...')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, {
                'name_list': name_list,
                'length_list': length_list,
                'data_dict': data_dict})

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def parse_text_line(self, line):
        raise NotImplementedError("Subclasses must implement parse_text_line")

    def load_obj_features(self, text_data):
        return {}

    def format_return(self, base_sample, obj_features):
        raise NotImplementedError("Subclasses must implement format_return")

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def inv_transform_th(self, data):
        return data * torch.from_numpy(self.std).to(
                data.device) + torch.from_numpy(self.mean).to(data.device)

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        key = self.name_list[idx]
        data = self.data_dict[key]

        motion, scale_betas, m_length, text_list = data['motion'], data['scale_betas'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)

        caption, tokens, species = text_data['caption'], text_data['tokens'], text_data['species']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # random cut long seq
        if m_length > self.max_motion_length:
            start_idx = random.randint(0, m_length - self.max_motion_length)
            motion = motion[start_idx:start_idx + self.max_motion_length]
            m_length = self.max_motion_length

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        # Crop fixed_len
        original_length = None
        if self.opt.fixed_len > 0:
            original_length = m_length
            m_length = self.opt.fixed_len

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        length = (original_length, m_length) if self.opt.fixed_len > 0 else m_length

        base_sample = {
            'word_embeddings': word_embeddings,
            'pos_one_hots': pos_one_hots,
            'caption': caption,
            'sent_len': sent_len,
            'motion': motion,
            'scale_betas': scale_betas,
            'length': length,
            'species': species,
            'tokens': '_'.join(tokens),
        }
        obj_features = self.load_obj_features(text_data)
        return self.format_return(base_sample, obj_features)


class Text2MotionDataset(MotionDataset):
    def parse_text_line(self, line):
        species, caption, tokens, *_ = line.strip().split('#')
        tokens = tokens.split(' ')
        return {
            'caption': ' '.join(caption.split()), # prevent multi space
            'tokens': [token for token in tokens if token.strip() != '' and token != '/SPACE'],
            'species': species,
        }

    def format_return(self, base_sample, obj_features):
        return (
            base_sample['word_embeddings'],
            base_sample['pos_one_hots'],
            base_sample['caption'],
            base_sample['sent_len'],
            base_sample['motion'],
            base_sample['scale_betas'],
            base_sample['length'],
            base_sample['species'],
            base_sample['tokens'],
        )


class TextObj2MotionDataset(MotionDataset):
    def parse_text_line(self, line):
        loc, objid, species, caption, tokens, *_ = line.strip().split('#')
        tokens = tokens.split(' ')
        return {
            'caption': ' '.join(caption.split()),
            'tokens': [token for token in tokens if token.strip() != '' and token != '/SPACE'],
            'species': species,
            'obj_id': objid,
            'loc': loc,
        }

    def load_obj_features(self, text_data):
        objid = text_data['obj_id']
        obj_BPS = np.load(pjoin(self.opt.obj_dir, objid, 'BPS.npz'))
        obj_sample_points = np.load(pjoin(self.opt.obj_dir, objid, 'sample_points.npy'))

        return {
            'obj_id': objid,
            'obj_BPS': obj_BPS['obj_BPS'],
            'obj_center_height': obj_BPS['obj_center_height'],
            'obj_scale': obj_BPS['obj_scale'],
            'obj_sample_points': obj_sample_points,
            'loc': text_data['loc'],
        }

    def format_return(self, base_sample, obj_features):
        return (
            base_sample['word_embeddings'],
            base_sample['pos_one_hots'],
            base_sample['caption'],
            base_sample['sent_len'],
            base_sample['motion'],
            base_sample['scale_betas'],
            base_sample['length'],
            base_sample['species'],
            base_sample['tokens'],
            obj_features['obj_id'],
            obj_features['obj_BPS'],
            obj_features['obj_center_height'],
            obj_features['obj_scale'],
            obj_features['obj_sample_points'],
            obj_features['loc'],
        )


class DogML3D(data.Dataset):
    def __init__(self, name, mode, split="train", **kwargs):

        self.mode = mode
        self.dataset_name = name

        print('Loading dataset %s ...' % self.dataset_name)
        opt = get_opt(self.dataset_name)
        self.opt = opt

        self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        self.w_vectorizer = WordVectorizer(opt.glove_dir, 'our_vocab')
        self.inner_dataset = Text2MotionDataset(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)

        assert len(self.inner_dataset) > 1

    def __getitem__(self, item):
        return self.inner_dataset.__getitem__(item)

    def __len__(self):
        return self.inner_dataset.__len__()


class DogML3D_Object(data.Dataset):
    def __init__(self, name, mode, split="train", **kwargs):
        self.mode = mode
        self.dataset_name = name

        print('Loading dataset %s ...' % self.dataset_name)
        opt = get_opt(self.dataset_name)
        self.opt = opt

        self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        self.w_vectorizer = WordVectorizer(opt.glove_dir, 'our_vocab')
        self.inner_dataset = TextObj2MotionDataset(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)

        assert len(self.inner_dataset) > 1

    def __getitem__(self, item):
        return self.inner_dataset.__getitem__(item)

    def __len__(self):
        return self.inner_dataset.__len__()
