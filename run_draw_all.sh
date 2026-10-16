#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${CONFIG_DIR:-$REPO_ROOT/configs/evals/fine-grain-ablation/draw}"
WORKDIR="${WORKDIR:-$REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-cuda:0}"

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "[ERROR] 配置目录不存在: $CONFIG_DIR" >&2
  exit 1
fi

cd "$WORKDIR"

mapfile -t YAML_FILES < <(find "$CONFIG_DIR" -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) | sort)

if [[ ${#YAML_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] 未找到 YAML 配置文件: $CONFIG_DIR" >&2
  exit 1
fi

echo "[INFO] 将按顺序执行 ${#YAML_FILES[@]} 个配置文件，device=$DEVICE"

for fname in "${YAML_FILES[@]}"; do
  echo "[RUN ] $fname"
  "$PYTHON_BIN" -m evals.draw_main \
    --fname "$fname" \
    --device "$DEVICE"
  echo "[DONE] $fname"
done

echo "[INFO] 全部 draw 配置执行完成。"