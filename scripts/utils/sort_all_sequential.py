#!/usr/bin/env python3
"""
对 all_sequential.txt 按「先 scene、再 bin 序号」从小到大排序。
每行格式: scene2013_05_28_drive_0000_sync_bin10195.pkl
"""
import argparse
import re


def sort_key(line: str):
    """提取 (scene名, bin序号) 作为排序键，scene 按字符串，bin 按整数。"""
    line = line.strip()
    if not line:
        return ("", 0)
    # 格式: scene..._sync_bin12345.pkl
    m = re.match(r"(.+)_bin(\d+)\.pkl$", line)
    if not m:
        return (line, 0)
    scene = m.group(1)  # e.g. scene2013_05_28_drive_0000_sync
    bin_num = int(m.group(2))
    return (scene, bin_num)


def main():
    parser = argparse.ArgumentParser(description="Sort all_sequential.txt by scene then bin index.")
    parser.add_argument(
        "input",
        nargs="?",
        default="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/all_sequential.txt",
        help="Input file path",
    )
    parser.add_argument(
        "-o", "--output",
        default="/home/zliu/IROS2026/Diff-StereoSplat/filenames/kitti360/train_complete/all_sequential_V2.txt",
        help="Output file (default: overwrite input)",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = [ln for ln in f if ln.strip()]

    lines_sorted = sorted(lines, key=sort_key)
    out_path = args.output or args.input

    with open(out_path, "w") as f:
        for ln in lines_sorted:
            f.write(ln if ln.endswith("\n") else ln + "\n")

    print(f"Sorted {len(lines_sorted)} lines -> {out_path}")


if __name__ == "__main__":
    main()
