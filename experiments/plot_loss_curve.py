import os

import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件（默认使用仓库内的实验日志，可用环境变量覆盖）
csv_file = os.environ.get('V2VA_LOSS_CSV', 'experiments/jepa_r0.csv')
data = pd.read_csv(csv_file)

# 检查数据列
print("CSV Columns:", data.columns)

# 确保包含 'step' 和 'loss' 列
if 'step' in data.columns and 'loss' in data.columns:
    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(data['step'], data['loss'], label='Loss', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.environ.get('V2VA_LOSS_PNG', 'figures/loss_curve.png'))  # 保存图像
    plt.show()
else:
    print("Error: CSV file does not contain 'step' and 'loss' columns.")