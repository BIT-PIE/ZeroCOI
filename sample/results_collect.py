import os
import numpy as np

def init_result_buffers():
    return {
        'motions': [],
        'verts': [],
        'lengths': [],
        'text': [],
        'species': [],
        'obj_ids': [],
        'obj_transform_trans': [],
        'obj_transform_rotation': [],
        'obj_scales': [],
    }


def append_rep_results(buffers, cond_mode, model_kwargs, joints, verts):

    buffers['text'] += model_kwargs['y']['text']
    buffers['species'] += model_kwargs['y']['species']
    buffers['motions'].append(joints.cpu().numpy())
    buffers['verts'].append(verts.cpu().numpy())
    buffers['lengths'].append(model_kwargs['y']['lengths'].cpu().numpy())

    if cond_mode == 'text_obj':
        buffers['obj_ids'] += model_kwargs['y']['obj_id']
        buffers['obj_transform_trans'].append(model_kwargs['y']['obj_transform_trans'].cpu().numpy())
        buffers['obj_transform_rotation'].append(model_kwargs['y']['obj_transform_rotation'].cpu().numpy())
        buffers['obj_scales'].append(model_kwargs['y']['obj_scales'].cpu().numpy())


def finalize_results(buffers, cond_mode):
    results = {
        'motion': np.concatenate(buffers['motions'], axis=0),
        'verts': np.concatenate(buffers['verts'], axis=0),
        'text': buffers['text'],
        'species': buffers['species'],
        'lengths': np.concatenate(buffers['lengths'], axis=0),
    }

    if cond_mode == 'text_obj':
        results.update({
            'obj_ids': buffers['obj_ids'],
            'obj_transform_trans': np.concatenate(buffers['obj_transform_trans'], axis=0),
            'obj_transform_rotation': np.concatenate(buffers['obj_transform_rotation'], axis=0),
            'obj_scales': np.concatenate(buffers['obj_scales'], axis=0),
        })

    return results


def get_render_sample(results, batch_size, sample_i, rep_i):
    data_idx = rep_i * batch_size + sample_i
    caption = results['text'][data_idx]
    species = results['species'][data_idx]
    length = results['lengths'][data_idx]
    verts = results['verts'][data_idx][:length]
    return data_idx, caption, species, verts


def get_render_obj_info(results, cond_mode, data_idx):
    if cond_mode != 'text_obj':
        return None, None, None, None

    return (
        results['obj_ids'][data_idx],
        results['obj_transform_trans'][data_idx],
        results['obj_transform_rotation'][data_idx],
        results['obj_scales'][data_idx],
    )


def save_results(out_path, args, results):
    os.makedirs(out_path, exist_ok=True)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    save_dict = {
        'motion': results['motion'],
        'verts': results['verts'],
        'text': results['text'],
        'species': results['species'],
        'lengths': results['lengths'],
        'num_repetitions': args.num_repetitions,
    }
    with open(npy_path.replace('.npy', '_text.txt'), 'w') as fw:
        fw.write('\n'.join(results['text']))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in results['lengths']]))

    if args.cond_mode == 'text_obj':
        save_dict.update({
            'obj_ids': results['obj_ids'],
            'obj_transform_trans': results['obj_transform_trans'],
            'obj_transform_rotation': results['obj_transform_rotation'],
            'obj_scales': results['obj_scales'],
        })
        with open(npy_path.replace('.npy', '_objid.txt'), 'w') as fw:
            fw.write('\n'.join(results['obj_ids']))

    np.save(npy_path, save_dict)
    return npy_path