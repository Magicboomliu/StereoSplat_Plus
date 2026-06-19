#!/usr/bin/env python3
"""Build a minimal KITTI360 demo dataset from a bin filelist.

Copies only files referenced by the listed feedforward_bins pkls and the
paths that the Stage2 dataloader derives from image paths (RGB, pseudo depth,
sparse lidar, metric depth). Layout mirrors the full KITTI360 tree.

Example:
  python scripts/data/build_kitti360_demo_dataset.py \
    --src /data1/StereoDatasets/KITTI/KITTI360 \
    --dst /data1/StereoDatasets/KITTI/KITTI360_demo \
    --filelist filenames/kitti360/trainval/demo_full.txt \
    --data-version bin_infos_8.0_FirstLIDAR \
    --pseudo-depth-type NMRFStereo
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path


def _read_bins(filelist: Path) -> list[str]:
    lines = [ln.strip() for ln in filelist.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _derived_paths(img_rel: str, pseudo_depth_type: str) -> list[str]:
    if "data_2d_raw" not in img_rel or not img_rel.endswith(".png"):
        return []
    out = [
        img_rel.replace("data_2d_raw", "monocular_depth/monodepthV2/data_2d_raw"),
        img_rel.replace("data_2d_raw", "projected_sparse_lidar/data_2d_raw"),
    ]
    if pseudo_depth_type == "Metric3DV2":
        dpt = img_rel.replace(
            "data_2d_raw", "monocular_depth/Metric3DV2/data_2d_raw"
        ).replace(".png", "_dpt.png")
        conf = dpt.replace("_dpt.npy", "_conf.png")
        out.extend([dpt, conf])
    elif pseudo_depth_type == "NMRFStereo":
        out.append(
            img_rel.replace("data_2d_raw", "PseudoDepth_NMRFStereo/data_2d_raw")
        )
    return out


def collect_relative_paths(
    src_root: Path,
    bins: list[str],
    data_version: str,
    pseudo_depth_type: str,
) -> set[str]:
    rel_paths: set[str] = set()
    for bin_name in bins:
        pkl_rel = Path("feedforward_bins") / data_version / bin_name
        pkl_abs = src_root / pkl_rel
        if not pkl_abs.exists():
            raise FileNotFoundError(pkl_abs)
        rel_paths.add(pkl_rel.as_posix())

        with open(pkl_abs, "rb") as f:
            bin_info = pickle.load(f)
        for sensor in ("CAM_LEFT", "CAM_RIGHT", "LIDAR_TOP"):
            for info in bin_info["sensor_info"][sensor]:
                data_path = info["data_path"]
                rel_paths.add(data_path)
                rel_paths.update(_derived_paths(data_path, pseudo_depth_type))
    return rel_paths


def copy_tree(
    src_root: Path,
    dst_root: Path,
    rel_paths: set[str],
    link: bool,
) -> tuple[int, int, list[str]]:
    copied = 0
    total_bytes = 0
    missing: list[str] = []
    for rel in sorted(rel_paths):
        src = src_root / rel
        dst = dst_root / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
        if link:
            os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)
        copied += 1
        total_bytes += src.stat().st_size
    return copied, total_bytes, missing


def write_manifest(dst_root: Path, bins: list[str], rel_paths: set[str], args: argparse.Namespace) -> None:
    manifest = dst_root / "demo_manifest.txt"
    lines = [
        f"src={args.src}",
        f"filelist={args.filelist}",
        f"data_version={args.data_version}",
        f"pseudo_depth_type={args.pseudo_depth_type}",
        f"bins={len(bins)}",
        f"files={len(rel_paths)}",
        "",
        "# bins",
        *bins,
        "",
        "# relative paths",
        *sorted(rel_paths),
    ]
    manifest.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/data1/StereoDatasets/KITTI/KITTI360"),
        help="Full KITTI360 root",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/data1/StereoDatasets/KITTI/KITTI360_demo"),
        help="Output mini dataset root",
    )
    parser.add_argument(
        "--filelist",
        type=Path,
        default=Path("filenames/kitti360/trainval/demo_full.txt"),
        help="Bin list (relative to repo stereosplat/ unless absolute)",
    )
    parser.add_argument(
        "--data-version",
        default="bin_infos_8.0_FirstLIDAR",
        help="feedforward_bins subfolder name",
    )
    parser.add_argument(
        "--pseudo-depth-type",
        default="NMRFStereo",
        choices=["NMRFStereo", "Metric3DV2", "MonocularDepthV2"],
        help="Must match training config depth_info_params.pseudo_depth_type",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink files instead of copying (faster, not standalone)",
    )
    parser.add_argument(
        "--unimatch-src",
        type=Path,
        default=Path(
            "/data1/zliu/feedforward_outputs_new/"
            "depth_estimation_224x840/checkpoint-90000/model.safetensors"
        ),
        help="UniMatch depth-estimator weights to bundle under pretrained/",
    )
    parser.add_argument(
        "--skip-unimatch",
        action="store_true",
        help="Do not copy UniMatch weights into the demo bundle",
    )
    parser.add_argument(
        "--stage1-src",
        type=Path,
        default=Path(
            "/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/"
            "Input_View_Invariant/withconf/stage1/latest/checkpoint-145000/"
            "model.safetensors"
        ),
        help="Stage1 init weights (model.safetensors only) to bundle under pretrained/",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Do not copy Stage1 weights into the demo bundle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be copied",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    filelist = args.filelist
    if not filelist.is_absolute():
        filelist = repo_root / filelist

    bins = _read_bins(filelist)
    rel_paths = collect_relative_paths(
        args.src, bins, args.data_version, args.pseudo_depth_type
    )

    print(f"[info] bins: {len(bins)}")
    print(f"[info] files to materialize: {len(rel_paths)}")
    if args.dry_run:
        for rel in sorted(rel_paths):
            print(rel)
        if not args.skip_unimatch and args.unimatch_src.exists():
            print("pretrained/depth_estimation_224x840/checkpoint-90000/model.safetensors")
        if not args.skip_stage1 and args.stage1_src.exists():
            print("pretrained/stage1/checkpoint-145000/model.safetensors")
        return

    args.dst.mkdir(parents=True, exist_ok=True)
    copied, total_bytes, missing = copy_tree(
        args.src, args.dst, rel_paths, link=args.link
    )

    def _bundle_pretrained(src: Path, rel: str) -> None:
        nonlocal copied, total_bytes
        if not src.exists():
            raise FileNotFoundError(src)
        dst = args.dst / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.link:
            os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)
        copied += 1
        total_bytes += src.stat().st_size
        rel_paths.add(rel)

    unimatch_rel = (
        "pretrained/depth_estimation_224x840/checkpoint-90000/model.safetensors"
    )
    if not args.skip_unimatch:
        _bundle_pretrained(args.unimatch_src, unimatch_rel)

    stage1_rel = "pretrained/stage1/checkpoint-145000/model.safetensors"
    if not args.skip_stage1:
        _bundle_pretrained(args.stage1_src, stage1_rel)

    write_manifest(args.dst, bins, rel_paths, args)

    readme = args.dst / "README.txt"
    readme.write_text(
        "Mini KITTI360 demo bundle for demo_full.txt quick iteration / cloud upload.\n"
        f"Source data: {args.src}\n"
        f"Bins: {len(bins)}\n"
        f"Files: {copied}\n"
        f"Size: {total_bytes / 1e6:.1f} MB\n"
        "\n"
        "Training paths (after unzip to same layout):\n"
        f"  datapath={args.dst}\n"
        f"  unimatch_weights_path={args.dst}/pretrained/"
        "depth_estimation_224x840/checkpoint-90000/model.safetensors\n"
        f"  stage_1_model_path={args.dst}/pretrained/"
        "stage1/checkpoint-145000/\n"
        "  train/val filelist=filenames/kitti360/trainval/demo_full.txt\n"
        "\n"
        "Note: only Stage1 model.safetensors is bundled (optimizer not needed for init).\n"
    )

    print(f"[done] copied: {copied}, missing: {len(missing)}, size: {total_bytes/1e6:.1f} MB")
    print(f"[done] dst: {args.dst}")
    if missing:
        raise SystemExit(f"Missing {len(missing)} files, first: {missing[:5]}")


if __name__ == "__main__":
    main()
