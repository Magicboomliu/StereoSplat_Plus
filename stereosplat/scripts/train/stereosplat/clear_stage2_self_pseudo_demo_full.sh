#!/usr/bin/env bash
# 清空 Self-Pseudo demo_full 训练的 work_dir 与 output_dir（与 train_stereosplat_stage2.sh 115-116 行一致）
set -euo pipefail

WORK_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/models/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_demo_full/"
OUTPUT_DIR="/data1/zliu/IROS26/Compared_With_Others_Pixi/train_visualization/stereosplat/Input_View_Invariant/withconf/stage2_self_pseudo_demo_full/"

EXPECTED_SUFFIX="stage2_self_pseudo_demo_full"

clear_dir() {
    local target="$1"
    if [[ "$target" != *"${EXPECTED_SUFFIX}"* ]]; then
        echo "[ERROR] 路径安全检查失败，拒绝删除: ${target}" >&2
        exit 1
    fi
    if [[ ! -d "$target" ]]; then
        mkdir -p "$target"
        echo "[OK] 目录不存在，已创建: ${target}"
        return
    fi
    find "$target" -mindepth 1 -delete
    echo "[OK] 已清空: ${target}"
}

echo "将清空以下两个目录内的全部内容（保留目录本身）:"
echo "  work_dir   = ${WORK_DIR}"
echo "  output_dir = ${OUTPUT_DIR}"
echo

clear_dir "$WORK_DIR"
clear_dir "$OUTPUT_DIR"

echo
echo "完成。"
