import gc
from typing import Any, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
import tinycudann as tcnn
from packaging import version

def config_to_primitive(config: Any, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg


def parse_version(ver: str):
    return version.parse(ver)


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, (int, float)):
        return value

    value = config_to_primitive(value)
    if not isinstance(value, list):
        raise TypeError("Scalar specification only supports list, got", type(value))

    if len(value) == 3:
        value = [0] + value
    assert len(value) == 4

    start_step, start_value, end_value, end_step = value
    if isinstance(end_step, int):
        current_step = global_step
        return start_value + (end_value - start_value) * max(
            min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
        )
    if isinstance(end_step, float):
        current_step = epoch
        return start_value + (end_value - start_value) * max(
            min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
        )
    return value


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()
