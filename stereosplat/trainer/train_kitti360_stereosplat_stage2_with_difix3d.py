import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math
import mmcv
import mmengine
from mmengine import MMLogger
from mmengine.config import Config
import logging
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import set_seed, convert_outputs_to_fp32, DistributedType, ProjectConfiguration, InitProcessGroupKwargs
import warnings
warnings.filterwarnings("ignore")
# torch.autograd.set_detect_anomaly(True)  # disabled: keeps all intermediates in memory, slows training ~30-50%
import sys
from pathlib import Path

# Inject difix3d/src so `import difix3d` works without pip-installing the package
# (mirrors the approach used in the validator scripts)
_STEREOSPLAT_ROOT = Path(__file__).resolve().parents[2]  # .../stereosplat
_DIFIX3D_SRC = _STEREOSPLAT_ROOT / "difix3d" / "src"
sys.path.insert(0, str(_STEREOSPLAT_ROOT))
if _DIFIX3D_SRC.is_dir():
    sys.path.insert(0, str(_DIFIX3D_SRC))

import numpy as np
from torch import Tensor,nn
from tools.metrics import saved_into_json,RGB_Quality_Meter,Depth_Quality_Meter
# from tools.metrics import RGB_Quality_Meter,Depth_Quality_Meter,saved_into_json
from mmengine.registry import MODELS
import random
from stereosplat.models_lab.StereoSplat.stereosplat import StereoSplat
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from tqdm import tqdm
import importlib

from difix3d import DifixRef
from eval.fusion_validation import (
    accumulate_batch_fusion_metrics,
    fusion_metric_from_accumulators,
    new_fusion_metric_accumulators,
)
from conf_tune_utils import (
    freeze_all_except_conf_modules,
    migrate_unified_to_split_conf_head,
)

def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def _load_state_dict_any(path: str, map_location: str = "cpu") -> dict:
    if path is None or str(path).strip() == "":
        raise ValueError("args.stage_1_model_path is empty.")

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
        raise FileNotFoundError(f"stage_1_model_path not found: {p}")

    if p.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as _safe_load_file
        except Exception as e:
            raise ImportError(
                "Loading .safetensors requires `safetensors`. "
                "Install it or provide a .pt/.pth/.bin checkpoint."
            ) from e
        state_dict = _safe_load_file(p, device=map_location)
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

    # Common wrappers from DDP / accelerate / custom save
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    state_dict = _strip_prefix_if_present(state_dict, "model.")
    return state_dict

def _is_accelerate_checkpoint(path: str) -> bool:
    """Return True if path is a full accelerate checkpoint (contains optimizer state).
    A plain Stage1 weights directory only has model.safetensors / .bin, no optimizer."""
    if not os.path.isdir(path):
        return False
    return any(
        os.path.exists(os.path.join(path, fname))
        for fname in ("optimizer.bin", "optimizer.safetensors", "random_states_0.pkl")
    )


def _resolve_accelerate_resume_path(cfg, args) -> str | None:
    """Return accelerate checkpoint dir to resume, or None."""
    if getattr(args, "resume_from", None):
        cfg.resume_from = args.resume_from
    resume_from = getattr(cfg, "resume_from", None)
    if not resume_from or str(resume_from).strip().lower() in ("", "none"):
        return None
    if resume_from != "latest":
        path = resume_from
    elif os.path.isdir(cfg.work_dir):
        dirs = [d for d in os.listdir(cfg.work_dir) if d.startswith("checkpoint")]
        if not dirs:
            return None
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = os.path.join(os.path.abspath(cfg.work_dir), dirs[-1])
    else:
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.path.abspath(cfg.work_dir), path)
    if not _is_accelerate_checkpoint(path):
        return None
    return path

def _make_side_dict(factory):
    return {"left": factory(), "right": factory()}

def _make_rgb_meter():
    return RGB_Quality_Meter(psnr=0.0, ssim=0.0)

def _make_depth_meter():
    return Depth_Quality_Meter(mae=0.0, mse=0.0)

def _make_stage_meters():
    # output rgb/depth for center/first/last + input depth (left/right)
    return {
        "rgb": {
            "center": _make_side_dict(_make_rgb_meter),
            "first": _make_side_dict(_make_rgb_meter),
            "last": _make_side_dict(_make_rgb_meter),
        },
        "depth": {
            "center": _make_side_dict(_make_depth_meter),
            "first": _make_side_dict(_make_depth_meter),
            "last": _make_side_dict(_make_depth_meter),
        },
        "input_depth": _make_side_dict(_make_depth_meter),
    }

def _update_stage_meters(stage_meters, rgb_dict, depth_dict, input_depth_dict):
    # rgb_dict: {center_view/first_view/last_view: {left/right: {psnr,ssim}}}
    # depth_dict: {center_view/first_view/last_view: {left/right: {mae,mse}}}
    view_key_map = {
        "center": "center_view",
        "first": "first_view",
        "last": "last_view",
    }
    for view_short, view_key in view_key_map.items():
        for side in ("left", "right"):
            stage_meters["rgb"][view_short][side].update(
                rgb_dict[view_key][side]["psnr"],
                rgb_dict[view_key][side]["ssim"],
            )
            stage_meters["depth"][view_short][side].update(
                mae=depth_dict[view_key][side]["mae"],
                mse=depth_dict[view_key][side]["mse"],
            )
    for side in ("left", "right"):
        stage_meters["input_depth"][side].update(
            mae=input_depth_dict["input_depth"][side]["mae"],
            mse=input_depth_dict["input_depth"][side]["mse"],
        )

def _finalize_meters(obj):
    # Convert meter objects to stats dicts in-place-compatible manner
    if isinstance(obj, dict):
        return {k: _finalize_meters(v) for k, v in obj.items()}
    if hasattr(obj, "get_stats"):
        return obj.get_stats()
    return obj

# Volume / CV branch losses are intermediate; omit from wandb & console.
_INTERMEDIATE_LOSS_NAMES = frozenset({
    'recon_vol', 'perceptual_vol', 'depth_abs_volume',
    'recon_cv', 'perceptual_cv', 'depth_abs_cv',
})


def _filter_display_logs(logs: dict) -> dict:
    return {
        k: v for k, v in logs.items()
        if not any(f'loss_{name}' in k for name in _INTERMEDIATE_LOSS_NAMES)
    }


_MV_GT_METRICS = frozenset({
    'recon_gs', 'perceptual_gs', 'depth_abs_gs', 'depth_est_loss',
    'recon_pixel_fused', 'perceptual_pixel_fused',
})
_MV_CONF_METRICS = frozenset({
    'conf_gs', 'conf_mv_abs', 'conf_2v_abs',
    'conf_comparative', 'conf_pick', 'conf_pick_accuracy',
    'soft_hard_pick_agree', 'soft_w_mv_mean',
    'conf_gs_mean', 'conf_gt_mean', 'conf_mv_mean', 'conf_mv_gt_mean',
    'conf_2v_mean', 'conf_2v_gt_mean',
})
_MV_MARGIN_METRICS = frozenset({
    'fusion_2v_margin', 'fusion_mv_margin', 'mv_margin',
    'fusion_2v_margin_center', 'fusion_2v_margin_last',
    'fusion_mv_margin_center', 'fusion_mv_margin_last',
    'mv_margin_center', 'mv_margin_last',
})
_MV_ANCHOR_METRICS = frozenset({
    '2v_floor_mv', '2v_ceiling_mv',
})


def _log_metric_core(suffix: str) -> str:
    name = suffix
    if name.startswith('loss_'):
        name = name[5:]
    if name.endswith('_w'):
        name = name[:-2]
    return name


def _mv_wandb_subgroup(suffix: str) -> str:
    core = _log_metric_core(suffix)
    if core in _MV_MARGIN_METRICS:
        return 'margin'
    if core in _MV_ANCHOR_METRICS:
        return 'anchor'
    if core in _MV_CONF_METRICS:
        return 'conf'
    if core in _MV_GT_METRICS:
        return 'gt'
    return 'other'


def _group_train_logs(logs: dict, view_num: int, total_loss: float) -> dict:
    """WandB only: train/2v vs train/mv; mv split into gt / conf / margin."""
    filtered = _filter_display_logs(logs)
    out = {}
    if view_num == 2:
        out['train/2v/loss_total'] = float(total_loss)
        for k, v in filtered.items():
            suffix = k[len('train/'):] if k.startswith('train/') else k
            out[f'train/2v/{suffix}'] = v
    else:
        out['train/mv/loss_total'] = float(total_loss)
        for k, v in filtered.items():
            suffix = k[len('train/'):] if k.startswith('train/') else k
            sub = _mv_wandb_subgroup(suffix)
            out[f'train/mv/{sub}/{suffix}'] = v
    return out

def _extract_wandb_metrics(prefix: str, results_dict: dict) -> dict:
    out = {}
    for view in ("center", "first", "last"):
        rgb_key = f"rgb_{view}"
        depth_key = f"depth_{view}"
        rgb = results_dict.get(rgb_key, {})
        depth = results_dict.get(depth_key, {})
        for side in ("left", "right"):
            rgb_stats = rgb.get(side, {})
            depth_stats = depth.get(side, {})
            if "psnr" in rgb_stats:
                out[f"{prefix}/{rgb_key}/{side}/psnr"] = float(rgb_stats["psnr"])
            if "ssim" in rgb_stats:
                out[f"{prefix}/{rgb_key}/{side}/ssim"] = float(rgb_stats["ssim"])
            if "mae" in depth_stats:
                out[f"{prefix}/{depth_key}/{side}/mae"] = float(depth_stats["mae"])
            if "mse" in depth_stats:
                out[f"{prefix}/{depth_key}/{side}/mse"] = float(depth_stats["mse"])

    input_depth = results_dict.get("input_depth", {})
    for side in ("left", "right"):
        stats = input_depth.get(side, {})
        if "mae" in stats:
            out[f"{prefix}/input_depth/{side}/mae"] = float(stats["mae"])
        if "mse" in stats:
            out[f"{prefix}/input_depth/{side}/mse"] = float(stats["mse"])
    return out

def create_logger(log_file=None, is_main_process=False, log_level=logging.INFO):
    if not is_main_process:
        return None
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if is_main_process else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if is_main_process else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if is_main_process else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

def sample_2_to_6(n=1, floats=False, rng=None):
    """Return 1 or n random values in 2..6.
       ints by default; set floats=True for [2,6).

       view_num=2 is sampled at 10% to reduce pure-GT iterations;
       view_num 3/4/5/6 share the remaining 90% equally (22.5% each).
    """
    _rng = rng if rng is not None else random
    if floats:
        return _rng.uniform(2, 6) if n == 1 else [_rng.uniform(2, 6) for _ in range(n)]
    else:
        _population = [2, 3, 4, 5, 6]
        _weights    = [0.10, 0.225, 0.225, 0.225, 0.225]
        def _sample_one():
            r = _rng.random()
            cumsum = 0.0
            for v, w in zip(_population, _weights):
                cumsum += w
                if r < cumsum:
                    return v
            return _population[-1]
        return _sample_one() if n == 1 else [_sample_one() for _ in range(n)]


def sample_3_to_6(n=1, floats=False, rng=None):
    """Sample view_num in {3,4,5,6} with equal probability (conf-only tuning)."""
    _rng = rng if rng is not None else random
    if floats:
        return _rng.uniform(3, 6) if n == 1 else [_rng.uniform(3, 6) for _ in range(n)]
    _population = [3, 4, 5, 6]
    return _rng.choice(_population) if n == 1 else [_rng.choice(_population) for _ in range(n)]


def _iter_rng(global_iter: int, salt: int = 0) -> random.Random:
    """Deterministic RNG from global_iter so all DDP ranks take identical branches."""
    return random.Random(int(global_iter) * 9973 + int(salt))


def _all_ranks_should_skip(accelerator: Accelerator, local_skip: bool) -> bool:
    """If any rank skips backward, all ranks must skip to avoid NCCL deadlock."""
    if accelerator.num_processes <= 1:
        return local_skip
    flag = torch.tensor([1 if local_skip else 0], device=accelerator.device, dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())

def main(args):
    
    
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir


    cfg.prompt = args.prompt

    # -------- Optional CLI overrides (if provided) --------
    # exp_name / output_dir
    if getattr(args, "exp_name", None):
        cfg.exp_name = args.exp_name
    if getattr(args, "output_dir", None):
        cfg.output_dir = args.output_dir

    # dataset-related overrides (mirror stereosplat configs)
    if not hasattr(cfg, "dataset_params") or cfg.dataset_params is None:
        cfg.dataset_params = dict()

    def _set_if_not_none(key: str, value):
        if value is not None:
            cfg.dataset_params[key] = value
            setattr(cfg, key, value)

    _set_if_not_none("datapath", getattr(args, "datapath", None))
    _set_if_not_none("train_filelist", getattr(args, "train_filelist", None))
    _set_if_not_none("val_filelist", getattr(args, "val_filelist", None))
    _set_if_not_none("test_filelist", getattr(args, "test_filelist", None))
    _set_if_not_none("sequence", getattr(args, "sequence", None))
    _set_if_not_none("data_version", getattr(args, "data_version", None))
    _set_if_not_none("supp_view_nums", getattr(args, "supp_view_nums", None))

    if getattr(args, "world_center", None) is not None:
        cfg.world_center = args.world_center

    if getattr(args, "unimatch_weights_path", None) is not None:
        cfg.unimatch_weights_path = args.unimatch_weights_path
    
    logger_mm = MMLogger.get_instance('mmengine', log_level='WARNING')
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.work_dir, 
        logging_dir=os.path.join(cfg.work_dir, 'logs')
    )
    tracker_enabled = bool(getattr(args, "use_wandb", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=("wandb" if tracker_enabled else None),
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    if tracker_enabled and accelerator.is_main_process:
        # Prefer CLI args (passed from train.sh). Fall back to defaults.
        wandb_project = getattr(args, "wandb_project", None) or "volumefusion"
        wandb_entity = getattr(args, "wandb_entity", None)
        wandb_mode = getattr(args, "wandb_mode", None)
        wandb_run_name = getattr(args, "wandb_run_name", None) or cfg.exp_name

        wandb_api_key = getattr(args, "wandb_api_key", None)
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        accelerator.init_trackers(
            project_name=wandb_project,
            # config=config,
            init_kwargs={
                "wandb": {
                    "name": wandb_run_name,
                    **({"entity": wandb_entity} if wandb_entity else {}),
                    **({"mode": wandb_mode} if wandb_mode else {}),
                },
            }
        )

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)
        
        
    def _dataset_module_for_world_center(world_center: str | None) -> str:
        # 统一从 stereosplat.data.<dataset>.dataloader 导入，避免依赖顶层 stereosplat.KITTI360_* 别名模块
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
        # 默认回退到 CenterCam_Ref
        return "stereosplat.data.KITTI360_CenterCam_Ref.dataloader"

    datasets = importlib.import_module(_dataset_module_for_world_center(getattr(cfg, "world_center", None)))

    #################### Dataset Configurration #############################
    dataset_config = cfg.dataset_params

    # configure logger
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    if not osp.exists(osp.dirname(log_file)):
        os.makedirs(osp.dirname(log_file),exist_ok=True)
    logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)
    if logger is not None:
        logger.info(f'Config:\n{cfg.pretty_text}')


    # generate datasets
    dataset = getattr(datasets, 
                        dataset_config.dataset_name)

    train_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":dataset_config.val_filelist,
        "test_filelist":dataset_config.val_filelist,
        "data_version":dataset_config.data_version,
        "resolution":dataset_config.resolution, 
        "split":"train",
        "sequence":dataset_config.sequence,
        "use_center":dataset_config.use_center,
        "use_first": dataset_config.use_first,
        "use_last": dataset_config.use_last,
        "supp_view_nums": dataset_config.supp_view_nums,
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model
    }

    val_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":dataset_config.val_filelist,
        "test_filelist":dataset_config.val_filelist,
        "data_version":dataset_config.data_version,
        "resolution":dataset_config.resolution, 
        "split":"val",
        "sequence":dataset_config.sequence,
        "use_center":dataset_config.use_center,
        "use_first": dataset_config.use_first,
        "use_last": dataset_config.use_last,
        "supp_view_nums": 3,
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model
    }

    # Define the dataloader
    train_dataset = dataset(**train_params)
    val_dataset = dataset(**val_params)

    train_dataloader = DataLoader(
        train_dataset, dataset_config.batch_size_train, shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val,
        pin_memory=True,
    )
    
    ############################ Stage 1 Pretrained Model Here : Frozen, Just for Creating Psuedo Views, Not Optimized ###########################################
    # Load a frozen Stage1 copy ONLY for 2-view distillation (view_num==2 steps).
    # Keeps the 2v branch anchored to Stage1 quality so pseudo views stay stable.
    # NOT used for fusion supervision — view_num>2 always uses the current model's
    # own 2v render as the fusion reference / pseudo-view source.
    frozen_2v_ref_model = None
    if args.stage_1_model_path is not None:
        accelerator.print(f"[2vDistill] loading frozen Stage1 as 2-view teacher from: {args.stage_1_model_path}")
        frozen_2v_ref_model = StereoSplat(backbone=cfg.model.backbone,
                                        neck=cfg.model.neck,
                                        costvolume_gs=cfg.model.costvolume_gs,
                                        volume_gs=cfg.model.volume_gs,
                                        losses_params=cfg.model.losses_params,
                                        camera_args=cfg.camera_args,
                                        dataset_params=cfg.dataset_params,
                                        use_checkpoint=cfg.use_checkpoint)
        sd_2vref = _load_state_dict_any(args.stage_1_model_path, map_location="cpu")
        incompatible = frozen_2v_ref_model.load_state_dict(sd_2vref, strict=True)
        accelerator.print(
            f"[2vDistill] loaded. missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        frozen_2v_ref_model.eval()
        for p in frozen_2v_ref_model.parameters():
            p.requires_grad_(False)

    frozen_stage_1_model = None
    if not args.self_pseudo:
        frozen_stage_1_model = StereoSplat(backbone=cfg.model.backbone,
                                        neck=cfg.model.neck,
                                        costvolume_gs=cfg.model.costvolume_gs,
                                        volume_gs=cfg.model.volume_gs,
                                        losses_params=cfg.model.losses_params,
                                        camera_args=cfg.camera_args,
                                        dataset_params=cfg.dataset_params,
                                        use_checkpoint=cfg.use_checkpoint)

        if args.stage_1_model_path is None:
            raise ValueError(
                "You must pass `--stage_1_model_path` (checkpoint dir or weights file) "
                "to load the frozen stage-1 model."
            )
        accelerator.print(f"[Stage1] loading frozen model from: {args.stage_1_model_path}")
        sd_stage1 = _load_state_dict_any(args.stage_1_model_path, map_location="cpu")
        incompatible = frozen_stage_1_model.load_state_dict(sd_stage1, strict=True)
        accelerator.print(
            f"[Stage1] loaded. missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        # Freeze stage-1 model (used only for pseudo view creation)
        frozen_stage_1_model.eval()
        for p in frozen_stage_1_model.parameters():
            p.requires_grad_(False)
    

    #### Pre-Trained Difix3D Model Here ###########################################

    pretrained_diffix_model = DifixRef(
        pretrained_name="nvidia/difix_ref",
        pretrained_path=args.pretrained_difix3d,
        timestep=args.timestep,
        mv_unet=args.use_ref,
        deterministic_vae_encode=args.deterministic_vae_encode,
        deterministic_scheduler_step=args.deterministic_scheduler_step,
    )

    pretrained_diffix_model.set_eval()

    for p in pretrained_diffix_model.parameters():
        p.requires_grad_(False)


    
    ############################ Stage 2 Psuedo-GT-Mix Training Model Here ###########################################
    
    # Define the Model/Optimizer/Schduler Here
    my_model = StereoSplat(backbone=cfg.model.backbone,
                                    neck=cfg.model.neck,
                                    costvolume_gs=cfg.model.costvolume_gs,
                                    volume_gs=cfg.model.volume_gs,
                                    losses_params=cfg.model.losses_params,
                                    camera_args=cfg.camera_args,
                                    dataset_params=cfg.dataset_params,
                                    use_checkpoint=cfg.use_checkpoint)

    if args.conf_tune:
        _resume_path_early = _resolve_accelerate_resume_path(cfg, args)
        if _resume_path_early is not None:
            accelerator.print(
                f"[ConfTune] accelerate resume detected ({_resume_path_early}); "
                "skip --init_ckpt load (will restore from checkpoint)."
            )
        else:
            init_path = args.init_ckpt
            if init_path is None:
                raise ValueError(
                    "--conf_tune requires --init_ckpt (Stage2 checkpoint for weight surgery)."
                )
            accelerator.print(f"[ConfTune] loading init checkpoint from: {init_path}")
            sd_init = _load_state_dict_any(init_path, map_location="cpu")
            sd_init = migrate_unified_to_split_conf_head(sd_init)
            incompatible = my_model.load_state_dict(sd_init, strict=False)
            accelerator.print(
                f"[ConfTune] load done. missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
            if incompatible.missing_keys:
                accelerator.print(
                    f"[ConfTune] missing (first 8): {incompatible.missing_keys[:8]}"
                )
        n_trainable, n_frozen, train_names = freeze_all_except_conf_modules(my_model)
        accelerator.print(
            f"[ConfTune] frozen={n_frozen:,} params, trainable={n_trainable:,} params"
        )
        for _tn in train_names:
            accelerator.print(f"[ConfTune]   trainable: {_tn}")

    # Self-bootstrap: weights-only init from --stage_1_model_path BEFORE
    # accelerator.prepare.  This sets model weights for the first launch.
    # A later accelerator.load_state(resume_from) will override both the model
    # weights AND restore optimizer / scheduler / global_iter when resuming.
    elif args.self_pseudo:
        if args.stage_1_model_path is not None:
            accelerator.print(f"[SelfPseudo] init student weights from: {args.stage_1_model_path}")
            sd_init = _load_state_dict_any(args.stage_1_model_path, map_location="cpu")
            incompatible = my_model.load_state_dict(sd_init, strict=True)
            accelerator.print(
                f"[SelfPseudo] init done. missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
        else:
            accelerator.print("[SelfPseudo] --stage_1_model_path not set; student starts from random weights.")

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f'Number of params: {n_parameters}')
    
    param_groups = [
        {"params": [], "lr": cfg.lr},             # 默认组
        {"params": [], "lr": cfg.lr * 0.01},      # 'pretrained' 组，lr_mult=0.01
    ]
    
    for name, param in my_model.named_parameters():
        if not param.requires_grad:
            continue
        if "pretrained" in name:
            param_groups[1]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)


    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.optimizer.weight_decay,betas=(0.9, 0.999))
    # learning rate scheme
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        1 / (cfg.warmup_steps*accelerator.num_processes),
        1,
        total_iters=cfg.warmup_steps*accelerator.num_processes,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_train_steps*accelerator.num_processes, eta_min=cfg.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, scheduler], milestones=[cfg.warmup_steps*accelerator.num_processes])


    # move to the accelerate
    my_model, optimizer, train_dataloader, val_dataloader, scheduler= accelerator.prepare(
        my_model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    if args.conf_tune:
        _conf_model = my_model.module if hasattr(my_model, "module") else my_model
        freeze_all_except_conf_modules(_conf_model)
        accelerator.print("[ConfTune] re-applied conf-only freeze after accelerator.prepare")
    
    if frozen_stage_1_model is not None:
        frozen_stage_1_model.to(accelerator.device)
        frozen_stage_1_model.eval()
    if frozen_2v_ref_model is not None:
        frozen_2v_ref_model.to(accelerator.device)
        frozen_2v_ref_model.eval()
    pretrained_diffix_model.to(accelerator.device)
    pretrained_diffix_model.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)

    # resume and load
    global_iter = 0
    first_epoch = 0
    resume_step = -1
    path = None
    

    # Potentially load in the weights and states from a previous save
    path = _resolve_accelerate_resume_path(cfg, args)
    if path is None and cfg.resume_from and str(cfg.resume_from).strip().lower() not in ("", "none"):
        _raw = cfg.resume_from if cfg.resume_from != "latest" else "(latest under work_dir)"
        _msg = (
            f"[Resume] {_raw} is not an accelerate checkpoint (no optimizer state). "
            "Skipping accelerator.load_state."
        )
        accelerator.print(_msg)
        if logger is not None:
            logger.info(_msg)

    if path:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(path, map_location='cpu', strict=False)
        global_iter = int(path.rstrip("/").split("/")[-1].split("-")[1])
        first_epoch = global_iter // num_update_steps_per_epoch
        resume_step = global_iter % num_update_steps_per_epoch
        if args.conf_tune:
            _conf_model = my_model.module if hasattr(my_model, "module") else my_model
            freeze_all_except_conf_modules(_conf_model)
            accelerator.print("[ConfTune] re-applied conf-only freeze after resume")
        if accelerator.is_main_process and logger is not None:
            logger.info(
                '[Resume] successfully resumed from %s -> iter=%d/%d',
                path, global_iter, cfg.max_train_steps)
    else:
        if accelerator.is_main_process and logger is not None:
            logger.info(
                '[Train] starting from scratch (iter=0/%d)', cfg.max_train_steps)

    epoch = first_epoch
    _resumed_global_iter = global_iter if path else None

    if accelerator.is_main_process and logger is not None:
        logger.info('work dir: %s', args.work_dir)
        logger.info('max_train_steps: %d', cfg.max_train_steps)
        if _resumed_global_iter is not None and resume_step > 0:
            logger.info(
                '[Resume] fast-forwarding %d dataloader batches before iter=%d',
                resume_step, _resumed_global_iter)
        

    # training along the iterations.
    print_freq = cfg.print_freq
    use_ddp = accelerator.num_processes > 1

    # ---- Step-0 validation: sanity-check the validation pipeline and record
    #      the baseline metrics (Stage1-init weights) before any training.
    _run_initial_val = True
    while global_iter <= cfg.max_train_steps:
        my_model.train()
        data_time_s = time.time()
        time_s = time.time()
        
        for i_iter, batch in enumerate(train_dataloader):
            if epoch == first_epoch and i_iter < resume_step:
                if (accelerator.is_main_process and logger is not None
                        and resume_step > 0
                        and (i_iter == 0 or (i_iter + 1) % 200 == 0
                             or i_iter + 1 == resume_step)):
                    logger.info(
                        '[Resume] skipping batch %d/%d (fast-forward)',
                        i_iter + 1, resume_step)
                continue

            data_time_e = time.time()

            # same view_num on all ranks (iter-seeded RNG)
            if args.conf_tune:
                sample_view_nums = sample_3_to_6(
                    n=1, floats=False, rng=_iter_rng(global_iter, salt=31))
            else:
                sample_view_nums = sample_2_to_6(
                    n=1, floats=False, rng=_iter_rng(global_iter, salt=31))
            if sample_view_nums == 2:
                view_num = 2
                matching_nums = 2
            elif sample_view_nums == 3:
                view_num = 3
                matching_nums = 3   # effective 4-view input (first+center)
            elif sample_view_nums == 4:
                view_num = 4
                matching_nums = 3
            elif sample_view_nums == 5:
                view_num = 5
                matching_nums = 5   # effective 6-view input (all pairs)
            elif sample_view_nums == 6:
                view_num = 6
                matching_nums = 5

            loss = torch.tensor(0.0, device=accelerator.device)
            logs = {}
            grad_norm = 0.0
            local_skip = False

            try:
                with accelerator.accumulate(my_model):
                    _fwd = my_model if not use_ddp else my_model.module
                    loss, logs, rendered_fusion_list, rendered_volume_list, rendered_cv_results_list = _fwd.forward_stage2_with_difix3d(
                        batch, "train",
                        view_num=view_num,
                        matching_nums=matching_nums,
                        iter=global_iter,
                        frozen_stage_1_model=frozen_stage_1_model,
                        pretrained_diffix_model=pretrained_diffix_model,
                        mix_psuedo_views_ratio=args.mix_psuedo_views_ratio,
                        mix_difix3d_ratio=args.mix_difix3d_ratio,
                        use_self_for_pseudo=args.self_pseudo,
                        frozen_2v_ref_model=frozen_2v_ref_model,
                        cfg=cfg)


                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    if torch.isnan(loss) or torch.isinf(loss):
                        if accelerator.is_main_process and logger is not None:
                            logger.warning(
                                '[Warning] NaN or INF loss at iter=%d/%d, skipping...',
                                global_iter, cfg.max_train_steps)
                        local_skip = True
                    else:
                        accelerator.backward(loss)

                        if accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    if accelerator.is_main_process and logger is not None:
                        logger.warning(
                            '[OOM] Skipping iter=%d/%d (view_num=%d) due to CUDA OOM.',
                            global_iter, cfg.max_train_steps, view_num)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    local_skip = True
                else:
                    raise e

            if _all_ranks_should_skip(accelerator, local_skip):
                optimizer.zero_grad(set_to_none=True)
                global_iter += 1
                continue

            _skip_periodic = (
                _resumed_global_iter is not None
                and global_iter == _resumed_global_iter)

            if (not _skip_periodic
                    and accelerator.sync_gradients and global_iter > 0
                    and cfg.save_freq > 0 and global_iter % cfg.save_freq == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), f'checkpoint-{global_iter}')
                    accelerator.save_state(save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest')
                    mmengine.utils.symlink(save_file_name, dst_file)
                    if logger is not None:
                        logger.info('[TRAIN] Save latest state dict to {}.'.format(save_file_name))
                accelerator.wait_for_everyone()

            _do_val = (not _skip_periodic
                       and accelerator.sync_gradients and global_iter > 0
                       and global_iter % cfg.val_freq == 0)
            # Step-1 baseline: run once at the very first update (effectively step-0
            # weights; one batch update is negligible). Skipped when resuming
            # mid-training (global_iter already >> 1 on first check).
            if not _do_val and _run_initial_val and global_iter <= 1:
                _do_val = True
            if _run_initial_val:
                _run_initial_val = False   # clear after first check regardless
            if _skip_periodic and accelerator.is_main_process and logger is not None:
                if global_iter % cfg.val_freq == 0 or (
                        cfg.save_freq > 0 and global_iter % cfg.save_freq == 0):
                    logger.info(
                        '[Resume] skip redundant save/val at iter=%d (already done)',
                        global_iter)
            if _do_val:
                if accelerator.is_main_process and logger is not None:
                    logger.info('[Val] iter=%d/%d validation start', global_iter,
                                cfg.max_train_steps)
                accelerator.wait_for_everyone()
                my_model.eval()
                if accelerator.is_main_process:
                    # Validation produces ONE json per step:
                    #   fusion_metric.json  — three sections:
                    #     "2view"            : pure 2-view GT (first stereo input)
                    #     "pseudo_multiview" : 6-view (2 GT + 4 pseudo), multiview render
                    #     "pseudo_fused"     : same input, pixel-wise conf fusion
                    #   Each section: psnr_first / psnr_center / psnr_last / psnr_mean

                    acc_2view, acc_pseudo_mv, acc_pseudo_fus = new_fusion_metric_accumulators()

                    overall_val_batch_save_dir = osp.join(
                        cfg.output_dir, cfg.exp_name, "validation",
                        "step-{}".format(global_iter))
                    os.makedirs(overall_val_batch_save_dir, exist_ok=True)

                    for i_iter_val, batch_val in enumerate(val_dataloader):
                        print("Processed {}/{}".format(i_iter_val, len(val_dataloader)))
                        val_batch_save_dir = os.path.join(
                            overall_val_batch_save_dir, "batch-{}".format(i_iter_val))
                        if getattr(cfg, 'validation_vis_progress', False):
                            os.makedirs(val_batch_save_dir, exist_ok=True)

                        accumulate_batch_fusion_metrics(
                            acc_2view,
                            acc_pseudo_mv,
                            acc_pseudo_fus,
                            my_model,
                            batch_val,
                            cfg,
                            frozen_stage_1_model=frozen_stage_1_model,
                            pretrained_diffix_model=pretrained_diffix_model,
                            self_pseudo=args.self_pseudo,
                            val_batch_save_dir=val_batch_save_dir,
                            save_visuals=getattr(cfg, 'validation_vis_progress', False),
                        )

                    fusion_metric = fusion_metric_from_accumulators(
                        acc_2view, acc_pseudo_mv, acc_pseudo_fus)

                    saved_into_json(
                        data_dict=fusion_metric,
                        path=os.path.join(overall_val_batch_save_dir, "fusion_metric.json"))

                    if tracker_enabled:
                        wandb_logs = {}
                        for _sec, _sec_data in fusion_metric.items():
                            for _k, _v in _sec_data.items():
                                if _v is not None:
                                    wandb_logs["val/{}/{}".format(_sec, _k)] = _v
                        accelerator.log(wandb_logs, step=global_iter)

                accelerator.wait_for_everyone()
                my_model.train()



            time_e = time.time()

            # print loss log regularly
            if global_iter % print_freq == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]['lr']
                losses_str = ""
                for loss_k, loss_v in _filter_display_logs(logs).items():
                    losses_str += ("%s: %.3f, " % (loss_k, loss_v))
                if logger is not None:
                    logger.info(
                        '[TRAIN] iter %5d/%d | Loss: %.3f, %s grad_norm: %.1f, '
                        'lr: %.7f, time: %.3f (%.3f)',
                        global_iter, cfg.max_train_steps, loss.item(), losses_str,
                        grad_norm, lr, time_e - time_s, data_time_e - data_time_s)

            global_iter += 1

            # wandb only: group by view_num==2 (train/2v) vs view_num>2 (train/mv)
            if tracker_enabled:
                accelerator.log(
                    _group_train_logs(
                        logs, view_num,
                        loss.item() if hasattr(loss, 'item') else float(loss)),
                    step=global_iter)

            data_time_s = time.time()
            time_s = time.time()

            if global_iter > cfg.max_train_steps:
                break

        if global_iter > cfg.max_train_steps:
            break

        epoch += 1

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
            
        
if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--work-dir', type=str, required=True)
    parser.add_argument('--resume-from', type=str, default='')
    
    # stereosplat stage 1 pre-trained model
    parser.add_argument('--stage_1_model_path', type=str, default=None)
    parser.add_argument(
        '--conf_tune',
        action='store_true',
        default=False,
        help='Conf-only fine-tuning: split conf_head, freeze RGB/geometry, fuse-only loss.',
    )
    parser.add_argument(
        '--init_ckpt',
        type=str,
        default=None,
        help='Stage2 checkpoint for conf_tune init (weight surgery from unified head).',
    )
    parser.add_argument('--mix_psuedo_views_ratio', type=float, default=0.5)
    parser.add_argument('--mix_difix3d_ratio', type=float, default=None,
                        help='Probability of applying Difix3D enhancement to pseudo views. '
                             'Defaults to mix_psuedo_views_ratio if not set.')
    # self-bootstrap: one unified model. The student loads --stage_1_model_path as
    # initialization and then generates pseudo views from its OWN current weights
    # (no separate frozen Stage1 model).
    parser.add_argument('--self_pseudo', action='store_true', default=False)
    
    
    # difix3d per-trained model
    parser.add_argument("--pretrained_difix3d",type=str,default=None)

    parser.add_argument('--timestep', type=int, default=199)
    parser.add_argument('--prompt', type=str, default="remove degradation")
    parser.add_argument('--use_ref', action='store_true', default=False)

    parser.add_argument('--deterministic_vae_encode', action='store_true', default=False)
    parser.add_argument('--deterministic_scheduler_step', action='store_true', default=False)
        
    

    # Optional overrides (take precedence over cfg file if provided)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    
    parser.add_argument('--datapath', type=str, default=None)
    parser.add_argument('--train-filelist', type=str, default=None)
    parser.add_argument('--val-filelist', type=str, default=None)
    parser.add_argument('--test-filelist', type=str, default=None)
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--data-version', type=str, default=None)
    parser.add_argument('--supp-view-nums', type=int, default=None)
    parser.add_argument(
        '--world-center',
        type=str,
        default=None,
        help='Center_LiDAR | First_Cam0 | First_LiDAR | First_LiDAR_3_Uniform',
    )
    parser.add_argument('--unimatch-weights-path', type=str, default=None)
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging via accelerate trackers',
    )
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument(
        '--wandb-mode',
        type=str,
        default=None,
        help='online | offline | disabled',
    )
    parser.add_argument(
        '--wandb-api-key',
        type=str,
        default=None,
        help='Optional. Prefer `wandb login` over passing key in args.',
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='If set, overrides wandb run name (otherwise uses cfg.exp_name).',
    )
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    
    
    main(args)