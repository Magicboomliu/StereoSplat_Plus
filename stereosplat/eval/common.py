"""Shared helpers for eval/run.py and legacy validator wrappers."""
from __future__ import annotations

import os
import os.path as osp
import sys
from pathlib import Path

import torch
from accelerate import Accelerator


def stereosplat_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_import_paths() -> Path:
    root = stereosplat_root()
    difix_src = root / "difix3d" / "src"
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if difix_src.is_dir():
        difix_str = str(difix_src)
        if difix_str not in sys.path:
            sys.path.insert(0, difix_str)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    return root


def config_path_for_stage(training_stage: str) -> str:
    root = stereosplat_root()
    if training_stage == "stage1":
        return str(
            root / "src/stereosplat/configs/stereosplat/input_invariant_stereosplat_default.py"
        )
    if training_stage == "stage2":
        return str(
            root / "src/stereosplat/configs/stereosplat/input_invariant_stereosplat_stage2.py"
        )
    raise ValueError(f"Unknown training_stage: {training_stage}")


def dataset_module_for_world_center(world_center: str | None) -> str:
    if world_center is None or world_center == "Center_LiDAR":
        return "stereosplat.data.KITTI360_CenterCam_Ref.dataloader"
    if world_center == "First_Cam0":
        return "stereosplat.data.KITTI360_FirstCam_Ref.dataloader"
    if world_center == "First_LiDAR":
        return "stereosplat.data.KITTI360_FirstLiDAR_Ref.dataloader"
    if world_center == "First_LiDAR_3_Uniform":
        return "stereosplat.data.KITTI360_FisrtLiDAR_Random.dataloader"
    if world_center == "First_Stage2":
        return "stereosplat.data.KITTI360_First_LiDAR_Random_Stage2.dataloader"
    return "stereosplat.data.KITTI360_CenterCam_Ref.dataloader"


def maybe_init_wandb(accelerator: Accelerator, args, cfg) -> bool:
    tracker_enabled = bool(getattr(args, "use_wandb", False))
    if not (tracker_enabled and accelerator.is_main_process):
        return False

    wandb_project = getattr(args, "wandb_project", None) or "StereoSplat"
    wandb_entity = getattr(args, "wandb_entity", None)
    wandb_mode = getattr(args, "wandb_mode", None)
    wandb_run_name = getattr(args, "wandb_run_name", None) or getattr(cfg, "exp_name", "validation")

    wandb_api_key = getattr(args, "wandb_api_key", None)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    accelerator.init_trackers(
        project_name=wandb_project,
        init_kwargs={
            "wandb": {
                "name": wandb_run_name,
                **({"entity": wandb_entity} if wandb_entity else {}),
                **({"mode": wandb_mode} if wandb_mode else {}),
            }
        },
    )
    return True


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_state_dict_any(path: str, map_location: str = "cpu") -> dict:
    if path is None or str(path).strip() == "":
        raise ValueError("Checkpoint path is empty.")

    p = os.path.expanduser(path)
    if osp.isdir(p):
        candidates = [
            "model.safetensors",
            "pytorch_model.bin",
            "pytorch_model.safetensors",
            "model.bin",
            "model.pt",
            "model.pth",
        ]
        chosen = None
        for c in candidates:
            cp = osp.join(p, c)
            if osp.exists(cp):
                chosen = cp
                break
        if chosen is None:
            raise FileNotFoundError(
                f"Could not find a model file in directory: {p}. "
                f"Tried: {', '.join(candidates)}"
            )
        p = chosen

    if not osp.exists(p):
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    if p.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load_file
        except Exception as e:
            raise ImportError(
                "Loading .safetensors requires `safetensors`. "
                "Install it or provide a .pt/.pth/.bin checkpoint."
            ) from e
        state_dict = safe_load_file(p, device=map_location)
    else:
        obj = torch.load(p, map_location=map_location)
        if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
            state_dict = obj["state_dict"]
        elif isinstance(obj, dict) and isinstance(obj.get("model"), dict):
            state_dict = obj["model"]
        elif isinstance(obj, dict):
            state_dict = obj
        else:
            raise ValueError(f"Unsupported checkpoint format at: {p} (type={type(obj)})")

    state_dict = _strip_prefix_if_present(state_dict, "module.")
    state_dict = _strip_prefix_if_present(state_dict, "model.")
    return state_dict
