# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import torch


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})
    
    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})
        
    if 'scale_betas' in notnone_batches[0]:
        scale_betas_batch = [b['scale_betas'] for b in notnone_batches]
        cond['y'].update({'scale_betas': torch.stack(scale_betas_batch)})
    
    if 'species' in notnone_batches[0]:
        species = [b['species'] for b in notnone_batches]
        cond['y'].update({'species': species})

    if 'bps_obj_feat' in notnone_batches[0]:
        bps_obj_feat_batch = [b['bps_obj_feat'] for b in notnone_batches]
        cond['y'].update({'bps_obj_feat': torch.stack(bps_obj_feat_batch)})
    
    if 'obj_id' in notnone_batches[0]:
        obj_ids = [b['obj_id'] for b in notnone_batches]
        cond['y'].update({'obj_id': obj_ids})
    
    if 'obj_center_height' in notnone_batches[0]:
        obj_center_height_batch = [b['obj_center_height'].squeeze() for b in notnone_batches]
        cond['y'].update({'obj_center_height': torch.stack(obj_center_height_batch)})
    
    if 'obj_scale' in notnone_batches[0]:
        obj_scale_batch = [b['obj_scale'].squeeze() for b in notnone_batches]
        cond['y'].update({'obj_scales': torch.stack(obj_scale_batch)})
    
    if 'obj_sample_points' in notnone_batches[0]:
        obj_sample_points_batch = [b['obj_sample_points'] for b in notnone_batches]
        cond['y'].update({'obj_sample_points': torch.stack(obj_sample_points_batch)})

    if 'loc' in notnone_batches[0]:
        locs = [b['loc'] for b in notnone_batches]
        cond['y'].update({'loc': locs}) 
    
    return motion, cond

# an adapter to text-motion collate func
def t2m_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2],
        'tokens': b[8],
        'lengths': b[6],
        'scale_betas': torch.tensor(b[5]).float(),
        'species': b[7]
    } for b in batch]
    return collate(adapted_batch)

# an adapter to text-obj-motion collate func
def t2m_obj_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1),
        'text': b[2],
        'tokens': b[8],
        'lengths': b[6],
        'scale_betas': torch.tensor(b[5]).float(),
        'species': b[7],
        'obj_id': b[9],
        'bps_obj_feat': torch.tensor(b[10]).float().unsqueeze(0),
        'obj_center_height': torch.tensor(b[11]).float(),
        'obj_scale': torch.tensor(b[12]).float(),
        'obj_sample_points': torch.tensor(b[13]).float(),
        'loc': b[14],
    } for b in batch]
    return collate(adapted_batch)

# an adapter to gt motion collate func
def t2m_eval_collate(batch):
    batch.sort(key=lambda x: x[3], reverse=True)

    word_embeddings = torch.stack([torch.from_numpy(b[0]) for b in batch], dim=0)
    pos_one_hots = torch.stack([torch.from_numpy(b[1]) for b in batch], dim=0)
    captions = [b[2] for b in batch]
    sent_lens = torch.tensor([b[3] for b in batch], dtype=torch.long)
    motions = collate_tensors([torch.from_numpy(b[4]) for b in batch])
    scale_betas = torch.stack([torch.from_numpy(b[5]) for b in batch], dim=0)

    lengths = torch.tensor([b[6] for b in batch], dtype=torch.long)

    return (
        word_embeddings,
        pos_one_hots,
        captions,
        sent_lens,
        motions,
        scale_betas,
        lengths,
    )