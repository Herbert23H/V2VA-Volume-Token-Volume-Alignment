import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 设置字体为 Times 风格，并为中文保留回退字体
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 表格数据
tasks = ['Organ', 'Nodule', 'Fracture', 'Adrenal', 'Vessel', 'Synapse']
methods = ['Average Pooling', 'Attentive Pooling', 'CLS Token', 'Token2Volume']

# 雷达半径缩放系数（<1 缩小，>1 放大），用于控制每个子图中雷达圈的视觉大小
RADAR_SCALE = 1

# Token2Volume 数值标签的径向外推量，数值越大，标签离图线越远
TOKEN2VOLUME_LABEL_OFFSET = 7.0

# Token2Volume 数值标签的切向偏移角度（弧度），用于避开外圈类别文字
TOKEN2VOLUME_LABEL_ANGLE_SHIFT = np.deg2rad(11)

# AUC 数据
# auc_data = {
#     'Average Pooling': [98.7, 81.2, 71.0, 87.4, 85.4, 78.8],
#     'Attentive Pooling': [98.5, 86.7, 73.2, 86.6, 82.2, 77.4],
#     'CLS Token': [99.0, 83.6, 71.5, 85.8, 87.4, 76.4],
#     'Token2Volume': [99.9, 94.8, 77.9, 91.3, 98.1, 94.6]
# }

# Precision 数据
precision_data = {
    'Average Pooling': [87.4, 70.2, 56.8, 72.3, 66.1, 75.8],
    'Attentive Pooling': [86.3, 78.0, 56.7, 73.2, 65.4, 72.7],
    'CLS Token': [89.4, 77.7, 54.2, 85.7, 65.7, 72.8],
    'Token2Volume': [97.5, 82.0, 67.4, 81.1, 94.0, 91.4]
}

# Recall 数据
recall_data = {
    'Average Pooling': [86.7, 68.7, 50.7, 59.6, 70.4, 61.4],
    'Attentive Pooling': [85.7, 77.0, 53.3, 66.1, 62.2, 60.8],
    'CLS Token': [88.3, 72.5, 46.5, 57.8, 76.9, 57.1],
    'Token2Volume': [97.4, 83.4, 57.6, 81.1, 91.3, 86.5]
}

def plot_radar(
    ax,
    data,
    categories,
    methods,
    title,
    colors,
    value_label_offset=TOKEN2VOLUME_LABEL_OFFSET,
    value_label_angle_shift=TOKEN2VOLUME_LABEL_ANGLE_SHIFT,
):
    """绘制单个雷达图"""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    ax = plt.subplot(1, 2, ax, projection='polar')
    # 缩放子图绘图区以减小雷达圈半径，同时保持子图居中
    try:
        pos = ax.get_position()
        new_h = pos.height * RADAR_SCALE
        delta_y = (pos.height - new_h) / 2.0
        ax.set_position([pos.x0, pos.y0 + delta_y, pos.width, new_h])
    except Exception:
        pass
    
    # 绘制每个方法
    for method, color in zip(methods, colors):
        values = data[method]
        values += values[:1]  # 闭合
        is_token2volume = method == 'Token2Volume'
        line_width = 2.0 if is_token2volume else 1.5
        marker_size = 6 if is_token2volume else 4
        ax.plot(angles, values, 'o-', linewidth=line_width, label=method, color=color, markersize=marker_size)
        if is_token2volume:
            ax.fill(angles, values, alpha=0.11, color=color, zorder=1)
            for angle, value in zip(angles[:-1], values[:-1]):
                # if 0 <= angle <= np.pi:
                #     label_angle = angle + value_label_angle_shift
                # else:
                #     label_angle = angle - value_label_angle_shift
                label_angle = angle + value_label_angle_shift
                # if 0 < angle < np.pi:
                #     horizontal_alignment = 'left'
                # elif np.pi < angle < 2 * np.pi:
                #     horizontal_alignment = 'right'
                # else:
                #     horizontal_alignment = 'center'
                horizontal_alignment = 'center'

                ax.text(
                    label_angle,
                    value + value_label_offset,
                    f'{value:.1f}',
                    color=color,
                    fontsize=14,
                    fontweight='bold',
                    ha=horizontal_alignment,
                    va='center',
                    # bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.75),
                    zorder=5,
                )
    
    # 设置标签和标题
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=14)
    ax.tick_params(axis='x', pad=20)
    ax.set_ylim(40, 100)
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_yticklabels(['', '', '', '', ''], size=14)
    ax.set_rlabel_position(0)
    ax.grid(True)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.8, color='#b0b0b0', alpha=0.3)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color='#b0b0b0', alpha=0.3)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_linestyle('--')
    ax.spines['polar'].set_linewidth(0.8)
    ax.spines['polar'].set_color('#b0b0b0')
    ax.spines['polar'].set_alpha(0.3)
    ax.set_title(title, size=16, weight='bold',pad=20)
    
    return ax

# 创建图表
fig = plt.figure(figsize=(11.5, 5.8))


# 颜色设置（Token2Volume突出）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Precision 雷达图
plot_radar(1, precision_data, tasks, methods, 'Precision(%)', colors)

# Recall 雷达图
plot_radar(2, recall_data, tasks, methods, 'Recall(%)', colors)
    
# 添加统一的图例：底部居中单行显示
handles = [plt.Line2D([0], [0], marker='o', color=color, markerfacecolor=color,
                       markersize=6, linewidth=2, label=method)
           for method, color in zip(methods, colors)]
fig.legend(handles=handles, labels=methods, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0.03), frameon=False, fontsize=14,
           handlelength=2.0, columnspacing=1.2, handletextpad=0.6,
           borderaxespad=0.0, title_fontsize=16,prop={"size":14, "weight":"bold"})


# 保留底部空间并微调两张子图的间距
# wspace 越大，两个子图之间的距离越大；bottom 控制图例所需留白
plt.subplots_adjust(left=0.00, right=1, top=0.88, bottom=0.19, wspace=0.01)
# 保存为矢量格式（可任意放大缩小，清晰度不变）
# plt.savefig('radar_comparison.svg', format='svg')
plt.savefig(os.environ.get("V2VA_RADAR_OUT", "figures/radar_comparison.pdf"), format='pdf')
# print("✓ 矢量图已保存为: radar_comparison.svg (SVG格式)")
print("✓ 矢量图已保存为: radar_comparison.pdf (PDF格式)")
plt.show()
