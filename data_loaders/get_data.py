# This code is based on https://github.com/GuyTevet/motion-diffusion-model
from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_obj_collate, t2m_eval_collate

def get_dataset_class(name):
    if name == 'dogml':
        from data_loaders.data.dataset import DogML3D
        return DogML3D
    elif name == 'dogml_obj':
        from data_loaders.data.dataset import DogML3D_Object
        return DogML3D_Object
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, mode='train'):
    if mode == 'gt':
        return t2m_eval_collate
    if name in ["dogml"]:
        return t2m_collate
    elif name in ["dogml_obj"]:
        return t2m_obj_collate
    else:
        return all_collate


def get_dataset(name, split='train', mode='train'):
    DATA = get_dataset_class(name)
    if name in ["dogml", "dogml_obj"]:
        dataset = DATA(name=name, split=split, mode=mode)
    else:
        dataset = DATA(split=split)
    return dataset


def get_dataset_loader(name, batch_size, split='train', mode='train'):
    dataset = get_dataset(name, split, mode)
    collate = get_collate_fn(name, mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate,
    )

    return loader