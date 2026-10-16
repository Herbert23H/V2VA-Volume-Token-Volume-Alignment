# ============================================================
# 常用命令备忘（以下路径均为仓库根目录下的相对路径）
# ============================================================
# 查当前是否设置了 OMP_NUM_THREADS 为空就表示当前 shell 没显式设置。
echo "$OMP_NUM_THREADS"
# 查所有线程相关环境变量
env | grep -E '^(OMP|MKL|OPENBLAS|NUMEXPR)_'
# 查机器可见 CPU 数
nproc
# 查当前进程可用 CPU 绑定范围（容器/调度器常影响这个）
taskset -cp $$

# 1.
screen -S mrnet
# 2.
conda activate jepa
cd /path/to/V2VA    # 仓库根目录，以下所有命令均以仓库根为工作目录
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=11
export NUMEXPR_NUM_THREADS=11

-----------------------------------------------------------------------------------
预训练
python -m app.main \
  --fname configs/pretrain/vitb16_384_medmnist_plus_3d_fine-grain.yaml \
  --devices cuda:0

------------------------------------------------------------------------------------
下游多分类（训练+验证）
python -m evals.main \
  --fname configs/evals/fine-grain-val/vitb16_384_medmnist_plus_3d_ad_sota.yaml \
  --devices cuda:0

------------------------------------------------------------------------------------

测试集（输出混淆矩阵）
python -m evals.onlytest_main \
  --fname configs/evals/fine-grain-val/vitb16_384_medmnist_plus_3d_ad_sota_test.yaml \
  --devices cuda:0

------------------------------------------------------------------------------------
分类头基准测试（params / FLOPs / 延迟）
python benchmark_heads.py --config-dir <你的 head 配置目录>

------------------------------------------------------------------------------------
绘图（独立于 eval/test）：PCA(before/after stem) + Grad-CAM(3-stage isosurface)
# src/datasets/medmnist3d_frozen.py line106:
# 'shuffle': True,    # 绘图方便找展示清晰图, 记得训练时改回training
# 换成你自己的 draw 配置
python -m evals.draw_main \
  --fname configs/evals/draw/<your_draw_config>.yaml \
  --device cuda:0

# 一键绘图（遍历 CONFIG_DIR 下所有 yaml）
CONFIG_DIR=configs/evals/draw bash run_draw_all.sh