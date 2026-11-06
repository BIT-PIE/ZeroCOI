import argparse
import re

import numpy as np
import torch
import pickle
import math
import sys
from pathlib import Path
from pytorch3d.transforms import matrix_to_euler_angles, rotation_6d_to_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
from utils.BITE_adapter import SMAL as BiteSMAL
import DigiDogs.digidogs as _digidogs
sys.modules.setdefault("digidogs", _digidogs)
import digidogs.rgbddog.reader as _digidogs_reader
from DigiDogs.digidogs.rgbddog.utils import ReadBvhFile as _official_read_bvh

_digidogs_reader.math = math


DATASETS = ("DigiDogs", "AniMo4D", "AnimalML3D")
ANIMO4D_DOG_SUBJECTS = {
    "african_wild_dog_female", "african_wild_dog_juvenile", "african_wild_dog_male",
    "arctic_fox_juvenile", "arctic_fox_male",
    "arctic_wolf_female", "arctic_wolf_juvenile", "arctic_wolf_male",
    "dhole_female", "dhole_juvenile", "dhole_male",
    "dingo_female", "dingo_juvenile", "dingo_male",
    "gray_wolf_female", "gray_wolf_juvenile", "gray_wolf_male",
    "maned_wolf_juvenile", "maned_wolf_male", "red_fox_juvenile",
    "spotted_hyena_female", "spotted_hyena_juvenile", "spotted_hyena_male",
    "striped_hyena_juvenile", "striped_hyena_male",
}
DIGIDOGS_SUBJECTS = {
    "coyote", "labrador", "poodle", "pug", "rottweiler", "shephard", "wilma", "wolf",
}
ANIMALML3D_SUBJECTS = {
    "doggieMN5", "foxWDFS", "foxXAT", "foxYSY", "foxZED", "huskydog3T",
}
SHAPE_PRIOR_PATH = PROJECT_ROOT / "BITE/data/smal_data/my_smpl_data_SMBLD_v3.pkl"


class InvalidTemplateScale(ValueError):
    pass


def read_digidogs_bvh(path):
    """Read official DigiDogs BVH data and restore offsets without editing the submodule."""
    points, connections, nodes = _official_read_bvh(path, gta=True)
    for frame in range(points.shape[2]):
        for index in range(1, len(nodes)):
            parent = connections[index]
            offset = np.asarray(nodes[index].offset, dtype=points.dtype)
            points[index, :, frame] = (
                points[parent, :, frame] + nodes[parent].rotation_world[frame] @ offset
            )
    return points, connections, nodes


def canonical_subject(dataset, name):
    if dataset == "AniMo4D":
        if name in ANIMO4D_DOG_SUBJECTS:
            species, gender = name.rsplit("_", 1)
            return f"{gender}_{species}"
        gender, separator, species = name.partition("_")
        raw_name = f"{species}_{gender}" if separator else ""
        return name if raw_name in ANIMO4D_DOG_SUBJECTS else None
    allowed = DIGIDOGS_SUBJECTS if dataset == "DigiDogs" else ANIMALML3D_SUBJECTS
    return name if name in allowed else None


def canonical_action(dataset, source_subject, action_name):
    action = action_name
    if dataset == "AniMo4D" and "__" in action:
        action = action.removesuffix("_keypoints.json").rsplit("_", 1)[-1]
    elif dataset == "AnimalML3D":
        prefix = source_subject + "_"
        if action.startswith(prefix):
            action = action[len(prefix):]
    elif dataset == "DigiDogs":
        for prefix in (f"DigiDogs_{source_subject}_", f"{source_subject}_"):
            if action.startswith(prefix):
                action = action[len(prefix):]
                break
    if not action or action.startswith("M"):
        raise ValueError(f"unsupported source action name: {action_name}")
    return action


def source_frame_count(path):
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim < 1:
            raise ValueError(f"{path}: motion array has no frame dimension")
        return len(array)
    if path.suffix == ".bvh":
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = re.match(r"\s*Frames:\s*(\d+)\s*$", line, re.I)
                if match:
                    return int(match.group(1))
        raise ValueError(f"{path}: BVH header has no Frames field")
    raise ValueError(f"unsupported motion file: {path}")


def segment_bounds(frames, max_frames):
    if frames < 2:
        raise ValueError("an action needs at least two frames")
    if frames <= max_frames:
        return [(0, frames)]
    bounds = [(start, min(start + max_frames, frames)) for start in range(0, frames, max_frames)]
    if bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-1] = (bounds[-2][1] - 1, frames)
    return bounds


class ShapePrior(torch.nn.Module):
    """Gaussian prior for SMAL shape coefficients.

    Adapted in part from https://github.com/benjiebob/WLDO and
    https://github.com/benjiebob/SMALify.
    """

    def __init__(self, prior_path):
        super().__init__()
        try:
            with open(prior_path, "r") as stream:
                values = pickle.load(stream)
        except (UnicodeDecodeError, TypeError):
            with open(prior_path, "rb") as stream:
                unpickler = pickle._Unpickler(stream)
                unpickler.encoding = "latin1"
                values = unpickler.load()

        betas_mean = values["dog_cluster_mean"]
        betas_cov = values["dog_cluster_cov"]
        inverse_cov = np.linalg.inv(betas_cov + 1e-5 * np.eye(betas_cov.shape[0]))
        self.register_buffer(
            "single_gaussian_precs",
            torch.tensor(np.linalg.cholesky(inverse_cov)).float(),
        )
        means = torch.tensor(betas_mean).float()
        self.register_buffer("single_gaussian_means", means)
        self.register_buffer("use_ind_tch", torch.from_numpy(np.ones(means.shape[0], dtype=bool)).float())

    def forward(self, betas_smal_orig, use_singe_gaussian=False):
        n_betas_smal = betas_smal_orig.shape[1]
        device = betas_smal_orig.device
        mask = self.use_ind_tch * torch.cat((
            torch.ones_like(self.use_ind_tch[:n_betas_smal]),
            torch.zeros_like(self.use_ind_tch[n_betas_smal:]),
        ))
        samples = torch.cat((
            betas_smal_orig,
            torch.zeros((betas_smal_orig.shape[0], self.single_gaussian_means.shape[0] - n_betas_smal)).float().to(device),
        ), dim=1)
        mean_sub = samples - self.single_gaussian_means.unsqueeze(0)
        precision = self.single_gaussian_precs * mask[:, None] * mask[None, :]
        result = torch.tensordot(mean_sub, precision, dims=([1], [0]))
        return torch.mean(result ** 2)


def source_entries(dataset_dir, dataset_name):
    """Return (animal, action, motion path, relative source path) for the official layouts."""
    dataset_dir = Path(dataset_dir)
    if dataset_name == "DigiDogs":
        text_dir = dataset_dir / "texts"
        if not text_dir.is_dir():
            raise FileNotFoundError(
                f"DigiDogs sequence metadata is missing: {text_dir}; "
                "ZeroCOI's released unmirrored text files are used to map sequence IDs"
            )
        sequence_targets = {}
        pattern = re.compile(r"^DigiDogs_([^_]+)_(.+?)(\d{3})(?:-(\d+))?$")
        for text_path in sorted(text_dir.glob("DigiDogs_*.txt")):
            match = pattern.match(text_path.stem)
            if not match or match.group(2).startswith("M") or match.group(4):
                continue
            animal, action, sequence = match.group(1), match.group(2) + match.group(3), match.group(3)
            if animal not in DIGIDOGS_SUBJECTS:
                continue
            if sequence in sequence_targets:
                raise ValueError(f"several DigiDogs base texts map sequence {sequence}")
            sequence_targets[sequence] = (animal, action)

        motions = {}
        for path in sorted(dataset_dir.rglob("MotionSequence.bvh")):
            sequence = path.parent.name
            if not sequence.isdigit():
                continue
            sequence = f"{int(sequence):03d}"
            if sequence in motions:
                raise ValueError(f"duplicate DigiDogs sequence directory {sequence}: {path}")
            motions[sequence] = path
        missing = sorted(set(sequence_targets) - set(motions))
        if missing:
            raise FileNotFoundError(
                f"DigiDogs MotionSequence.bvh is missing for {len(missing)} selected sequences: "
                + ", ".join(missing[:10])
            )
        return [
            (animal, action, motions[sequence], motions[sequence].relative_to(dataset_dir).as_posix())
            for sequence, (animal, action) in sorted(sequence_targets.items())
        ]

    entries = {}
    duplicates = 0
    for path in sorted(dataset_dir.rglob("*.npy")):
        stem = path.stem
        if dataset_name == "AniMo4D":
            source_subject, separator, _ = stem.partition("__")
            if not separator:
                continue
        else:
            source_subject, separator, _ = stem.partition("_")
            if not separator:
                continue
        animal = canonical_subject(dataset_name, source_subject)
        if animal is None:
            continue
        if dataset_name == "AnimalML3D":
            caption = dataset_dir.parent / "motion_captions" / stem / "screenshots.txt"
            if not caption.is_file() or not caption.read_text(encoding="utf-8-sig").strip():
                continue
        action = canonical_action(dataset_name, source_subject, stem)
        key = (animal, action)
        entry = (animal, action, path, path.relative_to(dataset_dir).as_posix())
        if key in entries:
            duplicates += 1
            continue
        entries[key] = entry
    if duplicates:
        print(
            f"{dataset_name}: collapsed {duplicates} source motions with the same animal/action "
            "name; the lexicographically first path is used"
        )
    return list(entries.values())


def collect_tasks(dataset_dir, dataset_name="AniMo4D", output_dir=None, max_frames=300):
    """Select a shape template per animal and batch-fit every official source motion."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported fitting dataset: {dataset_name}")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir or dataset_dir)
    fitted_dir = output_dir / "fitted"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    template_tasks, tasks = [], []
    by_subject = {}
    selected_entries = source_entries(dataset_dir, dataset_name)
    if not selected_entries:
        raise ValueError(f"no species action data in dataset directory: {dataset_dir}")
    for subject, action, path, source_motion_path in selected_entries:
        by_subject.setdefault(subject, []).append((action, path, source_motion_path))
    for subject, entries in sorted(by_subject.items()):
        species_tasks = []
        used_actions = set()
        for target_action, source_path, source_motion_path in sorted(entries):
            frames = source_frame_count(source_path)
            if frames < 2:
                print(
                    f"***** skip {dataset_name}/{subject}/{target_action}: "
                    f"motion has {frames} frame *****"
                )
                continue
            bounds = (segment_bounds(frames, max_frames)
                      if dataset_name == "DigiDogs" else [(0, frames)])
            for part, (start, end) in enumerate(bounds, 1):
                action_name = target_action if part == 1 else f"{target_action}-{part}"
                if dataset_name == "DigiDogs":
                    text_path = dataset_dir / "texts" / f"DigiDogs_{subject}_{action_name}.txt"
                    if not text_path.is_file() or not text_path.read_text(encoding="utf-8-sig").strip():
                        raise FileNotFoundError(
                            f"DigiDogs segment has no matching released text file: {text_path}"
                        )
                if action_name in used_actions:
                    raise ValueError(
                        f"several source motions map to {dataset_name}/{subject}/{action_name}"
                    )
                used_actions.add(action_name)
                species_tasks.append((
                    action_name,
                    str(source_path),
                    str(fitted_dir / f"{dataset_name}_{subject}_{action_name}.pkl"),
                    start,
                    end,
                    source_motion_path,
                ))
        if not species_tasks:
            continue
        existing_params = sorted(
            path for path in fitted_dir.glob(f"{dataset_name}_{subject}_*.pkl")
            if path.is_file()
        )
        if existing_params:
            template_path = str(existing_params[0])
            existing_action = existing_params[0].stem.removeprefix(
                f"{dataset_name}_{subject}_"
            )
            # an existing template needs no matching input or action file
            template_tasks.append((template_path, existing_action,
                                   None, template_path, 0, 0,
                                   existing_action))
        else:
            template_path = species_tasks[0][2]
            template_tasks.append((template_path, *species_tasks[0]))
        tasks.extend((template_path, *task) for task in species_tasks)
    if not template_tasks:
        print(f"***** no motions with at least two frames in {dataset_dir}; skipped *****")
    return template_tasks, tasks


def validate_template_scale(params, param_path):
    """The template's global scale must be finite with an absolute value within [0.1, 5]."""
    scale = float(params["scale"].detach().item())
    abs_scale = abs(scale)
    if math.isnan(scale):
        reason = "scale is NaN"
    elif abs_scale < 0.1:
        reason = f"abs(scale)={abs_scale} is below 0.1"
    elif abs_scale > 5:
        reason = f"abs(scale)={abs_scale} is above 5"
    else:
        return
    raise InvalidTemplateScale(
        f"invalid template shape parameters: {param_path}; scale={scale}; reason: {reason}"
    )


def prepare_templates(template_tasks, gpu_id, epochs_one, epochs_two, lr, dataset_name):
    """Reuse only templates already present at the target output path; fit the missing ones first and enter the batch stage only once all of them succeeded."""
    valid = []
    for (template_path, action_name, file_path, param_path, frame_start, frame_end,
         source_motion_path) in template_tasks:
        param_path_obj = Path(param_path)
        try:
            if not param_path_obj.is_file():
                if file_path is None:
                    raise FileNotFoundError(f"the selected output template does not exist: {param_path}")
                print(f"***** fit template: {file_path} -> {param_path} *****")
                fit_one_lbfgs(
                    template_path, file_path, param_path,
                    gpu_id, epochs_one, epochs_two, lr,
                    is_template=True, dataset_name=dataset_name,
                    frame_start=frame_start, frame_end=frame_end,
                    source_motion_path=source_motion_path,
                    action_name=action_name,
                )
            else:
                print(f"***** reuse template: {param_path} *****")
            if not param_path_obj.is_file():
                raise FileNotFoundError(f"template fitting produced no parameter file: {param_path}")
            params = load_params(param_path, False, "cpu")
            validate_template_scale(params, param_path)
        except InvalidTemplateScale as exc:
            param_path_obj.unlink(missing_ok=True)
            print(f"***** skip animal using invalid template: {exc} *****")
            continue
        valid.append((template_path, action_name, file_path, param_path,
                      frame_start, frame_end, source_motion_path))
    return valid


def get_optimizer(params, epochs, lr, is_first):
    if is_first:
        optimizer = torch.optim.LBFGS(
            [params["beta"], params["betas_limbs"], params["scale"], params["trans"], params["pose"]],
            lr=lr,
            max_iter=epochs,
            line_search_fn='strong_wolfe'
        )
    else:
        optimizer = torch.optim.LBFGS(
            [params["trans"], params["pose"]],
            lr=lr,
            max_iter=epochs,
            line_search_fn='strong_wolfe'
        )
    return optimizer


def get_target(dataset, file_path, device, frame_start=0, frame_end=None):
    ''' Note: npy files may not need conversion. '''
    if dataset=='AniMo4D':
        joints = np.load(file_path)
        if joints.shape[1]==30:
            target, confidence = get_animo4d_30_target(file_path, device)
        elif joints.shape[1]==35:
            target, confidence = get_animo4d_35_target(file_path, device)
        else:
            raise ValueError("***** AniMo4D file_path error! *****")
    elif dataset=='DigiDogs':
        if file_path.endswith(".npy"):
            target, confidence = get_digidogs_npy_target(file_path, device)
        elif file_path.endswith(".bvh"):
            target, confidence = get_digidogs_bvh_target(file_path, device)
        else:
            raise ValueError("***** DigiDogs file_path error! *****")
    elif dataset == 'AnimalML3D':
        joints = np.load(file_path, allow_pickle=False)
        if joints.ndim != 3 or joints.shape[1:] != (35, 3) or len(joints) < 2:
            raise ValueError(f"{file_path}: expected (T>=2, 35, 3), got {joints.shape}")
        if not np.isfinite(joints).all():
            raise ValueError(f"{file_path}: contains NaN or Inf")
        target = torch.from_numpy(joints.astype(np.float32, copy=False)).to(device)
        confidence = get_joint_confidence(device)
    else:
        raise ValueError("***** dataset_name error! *****")
    target = target[frame_start:frame_end]
    if len(target) < 2:
        raise ValueError(f"{file_path}: selected segment has fewer than two frames")
    return target, confidence


def fit_one_lbfgs(template_param_path, file_path, param_path, gpu_id, epochs_one,
                  epochs_two, lr, is_template=False, dataset_name="AniMo4D",
                  frame_start=0, frame_end=None, source_motion_path=None,
                  action_name=None):
    """Optimize one sample on a single GPU."""
    if not is_template and (template_param_path is None or not Path(template_param_path).is_file()):
        raise FileNotFoundError(f"batch fitting needs a saved species shape: {template_param_path}")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    target_joints, joint_confidence = get_target(
        dataset_name, file_path, device, frame_start, frame_end,
    )
    model = get_model("bite", device)  # pick the model
    frames = target_joints.shape[0]
    if is_template:  # optimize the template sample
        params = load_params(param_path, True, device, frames)
        params["pose"].register_hook(mask_joint_grad)  # reduces head jitter
        optimizer = get_optimizer(params, epochs_one, lr, True)
        train_lbfgs(model, params, optimizer, epochs_one, target_joints, joint_confidence, device, False)
        optimizer = get_optimizer(params, epochs_two, lr, True)
        train_lbfgs(model, params, optimizer, epochs_two, target_joints, joint_confidence, device, True)  # then retrain with temporal smoothing
        validate_template_scale(params, param_path)
    else:
        params = load_batch_shape_params(template_param_path, frames, device)
        params["pose"].register_hook(mask_joint_grad)
        optimizer = get_optimizer(params, epochs_one, lr, False)
        train_lbfgs(model, params, optimizer, epochs_one, target_joints, joint_confidence, device, False)
        optimizer = get_optimizer(params, epochs_two, lr, False)
        train_lbfgs(model, params, optimizer, epochs_two, target_joints, joint_confidence, device, True)  # then retrain with temporal smoothing
    params["_source_motion_path"] = source_motion_path or Path(file_path).name
    stem = Path(param_path).stem
    prefix = f"{dataset_name}_"
    suffix = f"_{action_name}" if action_name else ""
    if not stem.startswith(prefix) or not suffix or not stem.endswith(suffix):
        raise ValueError(f"cannot recover animal name from fitted result: {param_path}")
    params["_animal_name"] = stem[len(prefix):-len(suffix)]
    Path(param_path).parent.mkdir(parents=True, exist_ok=True)
    save_params(param_path, params)
    # free GPU memory; processes are isolated so no other process's memory is touched
    del model, optimizer, target_joints, params
    torch.cuda.empty_cache()


def train_lbfgs(model, params, optimizer, epochs, target_joints, joint_confidence, device, use_temporal_loss):
    """ lbfgs optimization routine """
    frames = target_joints.shape[0]
    shape_prior = ShapePrior(SHAPE_PRIOR_PATH).to(device)

    step_iter = 0
    def closure():
        nonlocal step_iter
        optimizer.zero_grad()
        _, joints = model(
            beta=params["beta"].repeat(frames, 1),
            betas_limbs=params["betas_limbs"].repeat(frames, 1),
            scale=torch.abs(params['scale']).reshape(1, 1).expand(frames, 1),
            trans=params["trans"],
            pose=cont6d_to_matrix(params["pose"]),
        )
        total_loss, joints_loss, temporal_loss = calculate_loss(
            joints, target_joints, joint_confidence, params, shape_prior, use_temporal_loss,
        )
        print(
            f"epoch {step_iter+1}/{epochs}: "
            f"total loss: {total_loss.item():.4f}",
            f"joints_loss: {joints_loss.item():.4f}",
            f"temporal_loss: {temporal_loss.item():.4f}",
        )
        total_loss.backward()
        step_iter += 1
        return total_loss

    optimizer.step(closure)


def fit_all(dataset_dir, gpu_id, epochs_one, epochs_two, lr,
            dataset_name="AniMo4D", output_dir=None, max_frames=300,
            *, on_templates_ready):
    """Fit one selected dataset sequentially on the supported single GPU."""
    template_tasks, tasks = collect_tasks(
        dataset_dir, dataset_name=dataset_name, output_dir=output_dir,
        max_frames=max_frames,
    )
    print(f"***** prepare {len(template_tasks)} species templates *****")
    template_tasks = prepare_templates(
        template_tasks, gpu_id, epochs_one, epochs_two, lr, dataset_name,
    )
    if not template_tasks:
        print(f"***** no valid {dataset_name} templates; skipped *****")
        return
    print("***** save validated template shapes *****")
    shape_paths = on_templates_ready(template_tasks, output_dir)
    valid_templates = {template_path for template_path, *_ in template_tasks}
    tasks = [
        (shape_paths[template_path], action_name, file_path, param_path,
         frame_start, frame_end, source_motion_path)
        for template_path, action_name, file_path, param_path,
            frame_start, frame_end, source_motion_path in tasks
        if template_path in valid_templates
    ]
    total = len(tasks)
    for index, (shape_path, action_name, file_path, param_path,
                frame_start, frame_end, source_motion_path) in enumerate(tasks, 1):
        print(f"***** fit [{index}/{total}]: {dataset_name}/{action_name} *****")
        fit_one_lbfgs(
            shape_path, file_path, param_path, gpu_id,
            epochs_one, epochs_two, lr,
            is_template=False, dataset_name=dataset_name,
            frame_start=frame_start, frame_end=frame_end,
            source_motion_path=source_motion_path,
            action_name=action_name,
        )
    print("***** all tasks finished *****")


def get_model(type, device):
    if type == "bite":
        return BiteSMAL(smal_model_type='39dogs_norm_newv3').to(device)  # the model a colleague used
    else:
        raise ValueError("model type error!")


def get_animo4d_30_target(file_path, device):
    origin_joints = np.load(file_path)
    target_joints = animo4d_30_to_smal_like(origin_joints, device)
    joint_confidence = get_joint_confidence(device)
    return target_joints, joint_confidence


def get_animo4d_35_target(file_path, device):
    target_joints = np.load(file_path)
    target_joints = torch.from_numpy(target_joints).to(device)
    joint_confidence = get_joint_confidence(device)
    return target_joints, joint_confidence


def get_digidogs_npy_target(file_path, device):
    ''' digidogs in npy format has already been converted to the SMAL skeleton '''
    target_joints = np.load(file_path)
    target_joints = torch.from_numpy(target_joints).to(device)
    joint_confidence = get_joint_confidence(device)  # animo4d's joint_confidence works here too
    return target_joints, joint_confidence


def get_digidogs_bvh_target(file_path, device):
    bvhPoints, _, _ = read_digidogs_bvh(file_path)
    bvh_joints = bvhPoints2Joints(bvhPoints)
    bvh_joints = bvh_joints[:, :, [1, 2, 0]]	# ZXY(up-axis is Y) -> XYZ
    bvh_joints[:, :, 1:2] -= np.min(bvh_joints[0, :, 1])	# move init pose to floor
    if bvh_joints.shape[1]==35:
        target_joints, joint_confidence = digidogs_35_to_smal_like(bvh_joints)
    else:
        target_joints, joint_confidence = digidogs_34_to_smal_like(bvh_joints)
    target_joints = torch.from_numpy(target_joints).float().to(device)
    joint_confidence = torch.from_numpy(joint_confidence).float().to(device)
    return target_joints, joint_confidence


def load_params(param_path, requires_grad, device, frames=0):
    """ Load the SMAL parameters from param_path; .pkl and .pth are supported. """
    metadata = {}
    if Path(param_path).exists():  # the parameter shapes are the same as in the else branch
        if param_path.endswith(".pkl"):
            with open(param_path, "rb") as f:
                params = pickle.load(f)
        elif param_path.endswith(".pth"):
            params = torch.load(param_path, map_location=device)
        else:
            raise ValueError(f"***** param_path type error: {param_path.suffix} *****")
        metadata = {key: params[key] for key in ("_source_motion_path", "_animal_name") if key in params}
        beta = torch.tensor(params["beta"], dtype=torch.float32, requires_grad=requires_grad, device=device)
        betas_limbs = torch.tensor(params["betas_limbs"], dtype=torch.float32, requires_grad=requires_grad, device=device)
        scale = torch.tensor(params["scale"], dtype=torch.float32, requires_grad=requires_grad, device=device)
        trans = torch.tensor(params["trans"], dtype=torch.float32, requires_grad=requires_grad, device=device)
        pose = torch.tensor(params["pose"], dtype=torch.float32, requires_grad=requires_grad, device=device)
        print(f"***** load params from {param_path} *****")
    else:  # everything defaults to torch.float32
        beta = torch.zeros((1, 30), requires_grad=requires_grad, device=device)
        betas_limbs = torch.zeros((1, 7), requires_grad=requires_grad, device=device)
        scale = torch.tensor(0.5, requires_grad=requires_grad, device=device)
        trans = torch.zeros((frames, 3), requires_grad=requires_grad, device=device)
        base = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        pose = base.repeat(frames, 35, 1)  # repeat returns a non-leaf tensor; it must become a leaf to be optimized
        pose = pose.clone().requires_grad_(requires_grad)  # 6d rotation representation, [frames, 35, 6]
        print("***** init params *****")
    result = {"beta": beta, "betas_limbs": betas_limbs, "scale": scale,
              "trans": trans, "pose": pose}
    result.update(metadata)
    return result


def load_batch_shape_params(shape_path, frames, device):
    """Load a saved species shape and initialize motion parameters for one action."""
    shape = np.load(shape_path, allow_pickle=False)
    if shape.shape != (38,) or not np.isfinite(shape).all():
        raise ValueError(f"invalid species shape file: {shape_path}; expected 38 finite values")
    shape = np.asarray(shape, dtype=np.float32)
    base = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
    return {
        "scale": torch.tensor(shape[0], dtype=torch.float32, device=device),
        "beta": torch.tensor(shape[1:31][None, :], dtype=torch.float32, device=device),
        "betas_limbs": torch.tensor(shape[31:][None, :], dtype=torch.float32, device=device),
        "trans": torch.zeros((frames, 3), requires_grad=True, device=device),
        "pose": base.repeat(frames, 35, 1).clone().requires_grad_(True),
    }


def save_params(param_path, params):
    """ Save the SMAL parameters to param_path (a pkl file). """
    metadata = {key: params[key] for key in ("_source_motion_path", "_animal_name") if key in params}
    with torch.no_grad():
        params = {
            "beta": params["beta"].cpu().numpy(),
            "betas_limbs": params["betas_limbs"].cpu().numpy(),
            "scale": params["scale"].cpu().numpy().item(),
            "trans": params["trans"].cpu().numpy(),
            "pose": params["pose"].cpu().numpy()
        }
        params.update(metadata)
    with open(param_path, "wb") as f:
        pickle.dump(params, f)
    print(f"***** save SMAL params to {param_path} *****")


def calculate_loss(joints, target_joints, joint_confidence, params, shape_prior, use_temporal_loss):
    """Compute the losses used by LBFGS fitting."""
    l2_loss = ((joints - target_joints) ** 2).sum(dim=-1)
    joints_loss = (l2_loss * joint_confidence).sum(dim=-1).mean()

    shape_prior_loss = torch.tensor(0.0)
    if shape_prior is not None:
        beta_prior_loss = shape_prior(params["beta"])
        limb_coeffs = torch.tensor([0.01, 1, 0.1, 1, 1, 0.1, 2], dtype=torch.float32, device=joints.device)
        limbs_prior_loss = torch.mean((params["betas_limbs"] * limb_coeffs[None, :]) ** 2)
        shape_prior_loss = 1e-5 * beta_prior_loss + limbs_prior_loss

    beta_penalty_loss = torch.mean(torch.relu(torch.abs(params["beta"]) - 1.0))
    beta_l2_loss = torch.mean(params["beta"] ** 2)
    limbs_penalty_loss = torch.mean(torch.relu(torch.abs(params["betas_limbs"]) - 0.65))
    beta_regular_loss = beta_penalty_loss + 1e-4 * beta_l2_loss + limbs_penalty_loss

    angle_penalty_loss = pose_angle_penalty_loss(params["pose"])
    theta = matrix_to_euler_angles(cont6d_to_matrix(params["pose"]), convention="ZXY")
    angle_l2_loss = torch.mean(torch.abs(theta[:, 2:]).sum(dim=-1).sum(dim=-1))
    angle_regulation_loss = angle_penalty_loss + 1e-4 * angle_l2_loss

    temporal_loss = get_temporal_loss(params["pose"], joints) * 10
    total_loss = joints_loss + shape_prior_loss + beta_regular_loss + angle_regulation_loss
    if use_temporal_loss:
        total_loss += temporal_loss
    return total_loss, joints_loss, temporal_loss


def cont6d_to_matrix(cont6d):
    """ Convert the 6d rotation representation into a rotation matrix.
    from pytorch3d.transforms import rotation_6d_to_matrix works too. """
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def pose_angle_penalty_loss(pose_rot6d):
    pose_matrix = cont6d_to_matrix(pose_rot6d) # [frame, N, 3, 3]
    theta = matrix_to_euler_angles(pose_matrix, convention='ZXY')
    # spine
    spine_skeleton = [2, 3, 4, 5, 6, 15, 16, 32]  # spine to head
    theta_min = torch.tensor([-torch.pi/15, -torch.pi/12, -torch.pi/12]).to(pose_rot6d.device)
    theta_max = torch.tensor([torch.pi/15, torch.pi/12, torch.pi/12]).to(pose_rot6d.device)
    spine_theta = theta[:, spine_skeleton, :]
    spine_angle_penatly_loss = torch.relu(theta_min - spine_theta) + torch.relu(spine_theta - theta_max)
    spine_angle_penatly_loss = spine_angle_penatly_loss.sum(dim=-1).sum(dim=-1).mean()
    # tail
    tail_skeleton = [25, 26, 27, 28, 29, 30, 31]
    theta_min = torch.tensor([-torch.pi/9, -torch.pi/4, -torch.pi/4]).to(pose_rot6d.device)
    theta_max = torch.tensor([torch.pi/9, torch.pi/4, torch.pi/4]).to(pose_rot6d.device)
    tail_theta = theta[:, tail_skeleton, :]
    tail_angle_penatly_loss = torch.relu(theta_min - tail_theta) + torch.relu(tail_theta - theta_max)
    tail_angle_penatly_loss = tail_angle_penatly_loss.sum(dim=-1).sum(dim=-1).mean()
    angle_penatly_loss = spine_angle_penatly_loss + tail_angle_penatly_loss
    return angle_penatly_loss


def get_temporal_loss(pose, joints):
    """ Keep consecutive frames smooth. """
    # temporal rotation smooth loss
    R = rotation_6d_to_matrix(pose.reshape((-1, 6))).reshape(-1, 34+1, 3, 3)  # pose
    R = R + 1e-6 * torch.randn_like(R)
    R_prev = R[:-1]  # (T-1, J, 3, 3)
    R_curr = R[1:]   # (T-1, J, 3, 3)
    # R_{t-1}^(-1) R_t
    R_rel = torch.einsum('tjik,tjkl->tjil', R_prev.transpose(-1, -2), R_curr)
    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)  # (T-1, J)
    cos_theta = (trace - 1) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    loss_theta = torch.mean(theta ** 2, dim=-1)
    # temporal 3D joints location smooth loss
    X_prev = joints[:-1]
    X_curr = joints[1:]
    diff = ((X_prev - X_curr) ** 2).sum(axis=2) + 1e-8  # the square can round to a negative value, which would make sqrt return nan
    loss_3d_location =  torch.mean(diff.sqrt(), dim=-1)
    return (loss_theta + loss_3d_location).mean()


def mask_joint_grad(grad):
    grad[:, 15:17, :] = 0  # neck, head
    grad[:, 32:, :] = 0  # mouth, ears
    return grad


def get_joint_confidence(device):
    """ Joint confidence (the loss weight; the larger the value, the more important) """
    return torch.tensor(  # used for animo4d
        [5, 5, 5, 5, 5, 5, 5,  # 0-6: root, pelvis0, spine, spine0, spine1, spine2, spine3
         1, 1, 1, 1,           # 7-10: LLeg1, LLeg2, LLeg3, LFoot
         1, 1, 1, 1,           # 11-14: RLeg1, RLeg2, RLeg3, RFoot
         1, 1,                 # 15-16: Neck, Head
         1, 1, 1, 1,           # 17-20: LLegBack1, LLegBack2, LLegBack3, LFootBack
         1, 1, 1, 1,           # 21-24: RLegBack1, RLegBack2, RLegBack3, RFootBack
         1, 1, 1, 1, 1, 1, 1,  # 25-31: Tail1, Tail2, Tail3, Tail4, Tail5, Tail6, Tail7
         0, 0, 0],              # 32-34: Mouth, LEar, REar
        dtype=torch.float32,
        device=device
    )


def bvhPoints2Joints(init_bvh_points):
    """ Turn the data read from a bvh file into joints. """
    bvh_points = init_bvh_points.transpose(2, 0, 1)     # [N, 3, frame] -> [frame, N, 3]
    num_joints = bvh_points.shape[1]
    joints = None
    # drop the end sites and reorder the joints
    if num_joints == 40:
        joints = np.concatenate([bvh_points[:, 0:9, :]] + [bvh_points[:, 10:16, :]] + [bvh_points[:, 17:23, :]] +
                                 [bvh_points[:, 24:29, :]] + [bvh_points[:, 30:35, :]] + [bvh_points[:, 36:39, :]], axis=1)
    elif num_joints == 41:      # the order is root to head, front legs, legs, tail
        joints = np.concatenate([bvh_points[:, 0:3, :]] + [bvh_points[:, 17:22, :]] + [bvh_points[:, 3:9, :]] + [bvh_points[:, 10:16, :]] +
                                 [bvh_points[:, 23:28, :]] + [bvh_points[:, 29:34, :]] + [bvh_points[:, 35:40, :]], axis=1)
    else:
        raise ValueError()
    return joints


def digidogs_35_to_smal_like(digidogs_joints):
    joints = np.zeros((digidogs_joints.shape[0], 35, 3)) # [frames, num_joints, 3]
    joint_confidence = np.ones(35)
    # root to head
    joints[:, [0, 3, 6, 15, 16, 32]] = digidogs_joints[:, [0, 1, 2, 4, 5, 7]]
    joints[:, 1] = joints[:, 0]
    joints[:, 2] = 0.5 * (joints[:, 1] + joints[:, 3])
    joints[:, 4] = 0.67 * joints[:, 3] + 0.33 * joints[:, 6]
    joints[:, 5] = 0.33 * joints[:, 3] + 0.67 * joints[:, 6]
    joints[:, [33, 34]] = joints[:, [16, 16]]
    joint_confidence[[1, 33, 34]] = 0.0
    joint_confidence[[2, 3, 4, 5, 32]] = 1e-1
    # tial
    joints[:, [25, 26, 28, 29, 31]] = digidogs_joints[:, [30, 31, 32, 33, 34]]
    joints[:, 27] = 0.5 * joints[:, 26] + 0.5 * joints[:, 28]
    joints[:, 30] = 0.5 * joints[:, 29] + 0.5 * joints[:, 31]
    joint_confidence[[27, 30]] = 1e-1
    # front leg
    joints[:, [7, 8, 9, 10]] = digidogs_joints[:, [9, 10, 11 ,13]]
    joints[:, [11, 12, 13, 14]] = digidogs_joints[:, [15, 16, 17, 19]]
    joint_confidence[[7, 8, 9, 11, 12, 13]] = 0.3
    # back leg
    joints[:, [17, 18, 19, 20]] = digidogs_joints[:, [20, 21, 22, 24]]
    joints[:, [21, 22, 23, 24]] = digidogs_joints[:, [25, 26, 27, 29]]
    joint_confidence[[17, 18, 19, 21, 22, 23]] = 0.3
    joint_confidence[[7, 11, 17, 21]] += 0.5
    return joints, joint_confidence


def digidogs_34_to_smal_like(digidogs_joints):
    joints = np.zeros((digidogs_joints.shape[0], 35, 3)) # [frames, num_joints, 3]
    joint_confidence = np.ones(35)
    # fix Neck0
    digidogs_joints[:, 5] = 0.5 * (digidogs_joints[:, 4] + digidogs_joints[:, 6])
    # root to head
    joints[:, [0, 2, 3, 5, 6, 15, 16, 32]] = digidogs_joints[:, [0, 1, 2, 3, 4, 5, 6, 8]]
    joints[:, 1] = joints[:, 0]  # interpolation
    joints[:, 4] = 0.5 * (joints[:, 3] + joints[:, 5])
    joints[:, [33, 34]] = joints[:, [16, 16]]
    joint_confidence[[1, 33, 34]] = 0.0
    joint_confidence[[2, 3, 4, 5, 32]] = 1e-1
    # tial
    joints[:, [26, 28, 31]] = digidogs_joints[:, [31, 32, 33]]
    joints[:, 25] = 0.5 * joints[:, 0] + 0.5 * joints[:, 26]
    joints[:, 27] = 0.5 * joints[:, 26] + 0.5 * joints[:, 28]
    joints[:, 29] = 0.67 * joints[:, 28] + 0.33 * joints[:, 31]
    joints[:, 30] = 0.33 * joints[:, 28] + 0.67 * joints[:, 31]
    joint_confidence[[27, 30]] = 1e-1
    # front leg
    joints[:, [7, 8, 9, 10]] = digidogs_joints[:, [10, 11, 12, 14]]
    joints[:, [11, 12, 13, 14]] = digidogs_joints[:, [16, 17, 18, 20]]
    joint_confidence[[7, 8, 9, 11, 12, 13]] = 0.3
    # back leg
    joints[:, [17, 18, 19, 20]] = digidogs_joints[:, [21, 22, 23, 25]]
    joints[:, [21, 22, 23, 24]] = digidogs_joints[:, [26, 27, 28, 30]]
    joint_confidence[[17, 18, 19, 21, 22, 23]] = 0.3
    joint_confidence[[7, 11, 17, 21]] += 0.5
    return joints, joint_confidence


def interpolate_points(p1, p2, n):
    """
    Interpolate n points between p1 and p2.
    p1, p2: (frames, 3)
    return: (frames, n, 3)
    """
    points = []
    for i in range(1, n+1):
        points.append(p1 + (p2 - p1) / (n + 1) * i)
    return np.stack(points, axis=1)


def animo4d_30_to_smal_like(animo4d_joints, device):
    frames = animo4d_joints.shape[0]
    print(f"***** shape: {animo4d_joints.shape} *****")
    joints = np.zeros((frames, 35, 3))
    # root -> spine3
    joints[:, 0] = animo4d_joints[:, 0]
    joints[:, 6] = animo4d_joints[:, 1]
    joints[:, 1:6] = interpolate_points(joints[:, 0], joints[:, 6], 5)
    # neck -> ear
    joints[:, 15] = animo4d_joints[:, 2]
    joints[:, 16] = animo4d_joints[:, 3]
    joints[:, 32] = animo4d_joints[:, 4]
    joints[:, 33] = animo4d_joints[:, 6]
    joints[:, 34] = animo4d_joints[:, 5]
    # tail1 -> tail7
    joints[:, 25] = animo4d_joints[:, 0]
    joints[:, 31] = animo4d_joints[:, 11]
    joints[:, 26:31] = interpolate_points(joints[:, 25], joints[:, 31], 5)
    # lleg1 -> lfoot
    joints[:, 7]  = animo4d_joints[:, 12]
    joints[:, 8]  = animo4d_joints[:, 13]
    joints[:, 9]  = animo4d_joints[:, 14]
    joints[:, 10] = animo4d_joints[:, 15]
    # rleg1 -> rfoot
    joints[:, 11] = animo4d_joints[:, 16]
    joints[:, 12] = animo4d_joints[:, 17]
    joints[:, 13] = animo4d_joints[:, 18]
    joints[:, 14] = animo4d_joints[:, 19]
    # LLegBack1  -> LFootBack
    joints[:, 17] = animo4d_joints[:, 20]
    joints[:, 18] = animo4d_joints[:, 21]
    joints[:, 19] = animo4d_joints[:, 23]
    joints[:, 20] = animo4d_joints[:, 24]
    # RLegBack1  -> RFootBack
    joints[:, 21] = animo4d_joints[:, 25]
    joints[:, 22] = animo4d_joints[:, 26]
    joints[:, 23] = animo4d_joints[:, 28]
    joints[:, 24] = animo4d_joints[:, 29]
    return torch.from_numpy(joints).float().to(device)


def load_shape_params(param_path):
    with Path(param_path).open("rb") as stream:
        params = pickle.load(stream)
    parts = []
    for key, size in (("scale", 1), ("beta", 30), ("betas_limbs", 7)):
        if key not in params:
            raise ValueError(f"{param_path}: missing parameter {key}")
        part = np.asarray(params[key], dtype=np.float32).reshape(-1)
        if part.size != size or not np.isfinite(part).all():
            raise ValueError(f"{param_path}: invalid {key}; expected {size} finite values")
        parts.append(np.abs(part) if key == "scale" else part)
    return np.concatenate(parts)


def save_template_shapes(template_tasks, output_dir):
    beta_dir = Path(output_dir) / "scale_betas"
    beta_dir.mkdir(parents=True, exist_ok=True)
    shape_paths = {}
    for template_path, _, _, param_path, _, _, _ in template_tasks:
        param_path = Path(param_path)
        with param_path.open("rb") as stream:
            subject = pickle.load(stream).get("_animal_name")
        if not subject:
            raise ValueError(f"{param_path}: missing _animal_name metadata")
        save_path = beta_dir / f"{subject}.npy"
        np.save(save_path, load_shape_params(param_path))
        shape_paths[str(template_path)] = str(save_path)
        print(f"{param_path} -> {save_path} (38,)")
    return shape_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit selected motions from one source dataset to the DSMAL model.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("dataset/DogML3D"), type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="single supported CUDA device")
    parser.add_argument("--shape-epochs", type=int, default=1000)
    parser.add_argument("--motion-epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-frames", type=int, default=300,
                        help="DigiDogs segment length; 300 reproduces the released DogML3D names")
    args = parser.parse_args(argv)
    if min(args.shape_epochs, args.motion_epochs, args.max_frames) < 1:
        parser.error("epoch counts and max-frames must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    fit_all(
        args.input_dir,
        args.gpu,
        args.shape_epochs,
        args.motion_epochs,
        args.learning_rate,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        on_templates_ready=save_template_shapes,
    )


if __name__ == "__main__":
    main()
