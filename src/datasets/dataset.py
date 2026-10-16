import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib  # 用于处理NIfTI格式医学图像
import pydicom  # 用于读取DICOM文件
from typing import Union, Optional
from monai.transforms import (
    Compose, RandFlipd, RandAffined,
    ScaleIntensityRanged, ToTensord, Resized, ScaleIntensityRangePercentilesd
)

# =====================================================================
# 2. 多标签膝关节MRI数据集定义
# =====================================================================
class KneeMRIDatasetV2MultiLabel(Dataset):
    """严格五模态膝关节MRI数据集，支持多标签分类（ACL/PCL/MCL/LCL损伤）

    特性：
    - 自动检查/生成NIfTI格式缓存
    - 保留DICOM原始像素值（不应用窗宽/窗位预处理）
    - 支持五模态数据并行加载
    - 提供数据完整性检查机制
    """

    # 定义五模态DICOM文件前缀（需与实际数据匹配）
    MODALITY_PREFIXES = [
        "pdcor",   # 模态1：PD加权冠状位
        "pdsag",  # 模态2：PD Dixon矢状位
        "pdtra",   # 模态3：PD加权横断位
        "t1sag",     # 模态4：T1加权矢状位
        "t2sag",       # 模态5：T2加权矢状位
    ]

    def __init__(
        self,
        root_dir: Union[str, Path],
        label_xlsx: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        transform=None,
        cache_only: bool = False,
    ) -> None:
        super().__init__()

        self.root_dir   = Path(root_dir).expanduser().resolve()
        self.cache_dir  = (Path(cache_dir).expanduser().resolve()
                           if cache_dir else self.root_dir.with_name(self.root_dir.name + "_nifti_cache"))
        self.transform  = transform
        self.cache_only = cache_only

        # ------------------------------------------------------------
        # ① 读取标签
        # ------------------------------------------------------------
        df = pd.read_excel(label_xlsx).dropna()
        df[["ACL", "PCL", "MCL", "LCL"]] = df[["ACL", "PCL", "MCL", "LCL"]].astype(int)
        self.label_map = {
            str(r["Patient ID"]).strip(): r[["ACL", "PCL", "MCL", "LCL"]].tolist()
            for _, r in df.iterrows()
        }

        # ------------------------------------------------------------
        # ② 收集“有标签”的病人（先不判断五模态）
        # ------------------------------------------------------------
        self.patients = [p.name for p in sorted(self.root_dir.glob("M*"))
                         if p.is_dir() and p.name in self.label_map]

        if not self.patients:
            raise RuntimeError("未找到任何带标签的病人目录，请检查根目录或 Excel。")

        # ------------------------------------------------------------
        # ③ 生成 / 校验 NIfTI 缓存
        # ------------------------------------------------------------
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.cache_only:
            self._ensure_cache()          # ➜ 缺啥补啥

        # ------------------------------------------------------------
        # ④ 再过滤一次：只留下五模态齐全的病人
        # ------------------------------------------------------------
        self.patients = [pid for pid in self.patients
                         if self._patient_has_all_modalities(pid)]

        if not self.patients:
            raise RuntimeError(
                "所有病人都缺少某些模态。\n"
                "请确认 MODALITY_PREFIXES 是否与实际 DICOM 目录名前缀一致，"
                "以及 DICOM→NIfTI 转换过程中是否报错。"
            )

    # -------------------- 核心接口方法 --------------------
    def __len__(self):
        """返回数据集样本数"""
        return len(self.patients)

    def __getitem__(self, idx):
        """加载单个样本的五模态数据和标签"""
        pid   = self.patients[idx]
        # 获取四分类标签并转为Tensor
        label = torch.tensor(self.label_map[pid], dtype=torch.long)

        # 加载五模态数据
        views = []
        for pref in self.MODALITY_PREFIXES:
            # 从缓存加载NIfTI文件
            nifti_path = self.cache_dir / pid / f"{pref}.nii.gz"
            try:
                img_np = nib.load(nifti_path).get_fdata(dtype=np.float32) # 确保float32类型
            except FileNotFoundError:
                raise RuntimeError(f"缓存缺失: {nifti_path}")  # 理论上不会发生
            img = torch.from_numpy(img_np).unsqueeze(0) # 添加通道维 [1, D, H, W]

            # 应用数据增强/变换
            if self.transform:
                img = self.transform({"image": img})["image"]
            views.append(img)

        return tuple(views), label
    
    # -------------------- 内部工具方法 --------------------
    def _patient_has_all_modalities(self, pid: str) -> bool:
        """检查患者是否具备全部五模态NIfTI文件"""
        p = self.cache_dir / pid
        return all((p / f"{pref}.nii.gz").exists() for pref in self.MODALITY_PREFIXES)

    def _ensure_cache(self):
        """缺多少补多少；已存在的 .nii.gz 不会重复转换"""
        """确保所有患者数据已转换为NIfTI格式"""
        for pid in self.patients:
            src = self.root_dir / pid          # 病人 DICOM 根
            dst = self.cache_dir / pid
            dst.mkdir(parents=True, exist_ok=True)

            for pref in self.MODALITY_PREFIXES:
                out_path = dst / f"{pref}.nii.gz"
                if out_path.exists():          # 已经缓存
                    continue

                # —— 找到对应 DICOM 目录（pref* 开头）——
                dcm_dir = next(src.glob(f"{pref}*"), None)
                if dcm_dir is None:
                    print(f"[警告] {pid} 缺失模态 {pref}，本轮无法生成 .nii.gz")
                    continue

                slices = self._read_series(dcm_dir)
                if slices is None:             # 空目录
                    continue

                vol = np.stack(slices, axis=-1)
                nib.save(nib.Nifti1Image(vol, affine=np.eye(4, dtype=np.float32)), out_path)
                

    # -------------------- DICOM处理相关 --------------------
    @staticmethod
    def _dicom_to_raw(ds: pydicom.Dataset):
        """从DICOM文件中提取原始像素值（不应用窗宽/窗位）"""
        return ds.pixel_array.astype(np.float32, copy=False)

    def _read_series(self, series_dir: Path):
        """读取DICOM序列并按InstanceNumber排序"""
        # 按切片编号排序
        files = sorted(
            series_dir.glob("*.dcm"),
            key=lambda f: int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber),
        )
        if not files:
            print(f"[空 DICOM 目录] {series_dir}")
            return None
        # 提取所有切片原始像素
        return [self._dicom_to_raw(pydicom.dcmread(f, force=True)) for f in files]

    def _convert_patient(self, src_dir: Path, dst_dir: Path, pid: str):
        """将单个患者的DICOM数据转换为NIfTI格式"""
        for pref in self.MODALITY_PREFIXES:
            out_path = dst_dir / f"{pref}.nii.gz"
            if out_path.exists():
                continue
                
            # 查找匹配的DICOM目录
            dcm_dir = next(src_dir.glob(f"{pref}*"), None)
            if not dcm_dir:
                print(f"[缺失模态] {pid} 无 {pref} DICOM")
                continue
                
            # 读取并保存为NIfTI
            slices = self._read_series(dcm_dir)
            if slices is None:
                continue
            vol = np.stack(slices, axis=-1)  # 沿第三维堆叠切片
            nib.save(nib.Nifti1Image(vol, affine=np.eye(4, dtype=np.float32)), out_path)

# =====================================================================
# 3. 数据变换配置
# =====================================================================
def get_monai_transform(spatial_size=(20, 256, 256)):
    """训练集数据增强管道（包含空间变换和强度归一化）"""
    return Compose([
        # 基于百分位的强度归一化（排除0.5%~99.5%外的异常值）
        ScaleIntensityRangePercentilesd(
            keys="image", 
            lower=0.5, 
            upper=99.5, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),  
        # 随机水平翻转（50%概率）
        RandFlipd(keys="image", spatial_axis=0, prob=0.5),
        # 随机仿射变换（旋转+缩放）
        RandAffined(
            keys="image", 
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1), 
            mode="bilinear", 
            prob=0.5
        ),
        # 调整空间尺寸（使用三线性插值）
        Resized(keys="image", spatial_size=spatial_size, mode="trilinear"),
        # 转换为PyTorch Tensor
        ToTensord(keys="image")
    ])

def get_monai_val_transform(spatial_size=(20, 256, 256)):
    """验证集变换管道（仅归一化和尺寸调整）"""
    return Compose([
        ScaleIntensityRangePercentilesd(
            keys="image", 
            lower=0.5, 
            upper=99.5, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        Resized(keys="image", spatial_size=spatial_size, mode="trilinear"),
        ToTensord(keys="image")
    ])

# 测试集通常与验证集使用相同变换
get_monai_test_transform = get_monai_val_transform

# =====================================================================
# 4. 数据加载工具函数
# =====================================================================
def custom_collate_fn(batch):
    """自定义批处理函数：合并五模态数据"""
    # batch结构：[( (view1,view2,...), label ), ...]
    views_batch, labels_batch = zip(*batch)
    
    # 按模态堆叠：生成5个[B, 1, D, H, W]的tensor
    stacked_views = [
        torch.stack([v[i] for v in views_batch], 0)
        for i in range(5)
    ]
    # 堆叠标签：[B, 4]
    labels = torch.stack(labels_batch, 0)
    return stacked_views, labels

import matplotlib.pyplot as plt

def validate_data_integrity(
        loader,
        tol_edge: float = 1e-3,
        tol_span: float = 0.8,
        max_samples: int = 1_000_000,   # 采样上限，防止显存／内存炸掉
        nbins: int = 100
    ):
    """
    数据完整性检查 + 直方图可视化
    ------------------------------------------------------------
    * NaN / Inf 检测
    * 归一化边界越界、跨度过窄检测
    * 全局 min / max 统计
    * 所有模态在同一画布上绘制直方图
    ------------------------------------------------------------
    参数：
        tol_edge : 边界允许的微小漂移
        tol_span : 要求每个样本覆盖的最小动态范围
        max_samples : 每个模态用于画图的最大采样数
        nbins : 直方图 bin 数
    """
    nan_inf_records     = []
    edge_violation      = []
    span_violation      = []
    value_range_records = {i: {"min": [], "max": []} for i in range(5)}
    hist_buckets        = [ [] for _ in range(5) ]      # 收集绘图用数据

    # ========== 遍历数据 ==========
    for batch_idx, (views_batch, _) in enumerate(loader):
        for mod_idx, modality in enumerate(views_batch):   # modality [B,1,D,H,W]
            B = modality.shape[0]
            flat = modality.view(B, -1)                    # [B, N]

            # ---- ① NaN / Inf ----
            nan_mask = torch.isnan(flat) | torch.isinf(flat)
            if nan_mask.any():
                bad = torch.nonzero(nan_mask.any(dim=1)).flatten().tolist()
                nan_inf_records.extend([(batch_idx, mod_idx, s) for s in bad])

            # ---- ② 归一化完整度 ----
            mins  = flat.min(dim=1).values
            maxs  = flat.max(dim=1).values
            spans = maxs - mins

            value_range_records[mod_idx]["min"].extend(mins.cpu().tolist())
            value_range_records[mod_idx]["max"].extend(maxs.cpu().tolist())

            bad_edge = ((mins < -tol_edge) | (maxs > 1 + tol_edge)).nonzero().flatten()
            edge_violation.extend([(batch_idx, mod_idx, int(i)) for i in bad_edge])

            bad_span = (spans < tol_span).nonzero().flatten()
            span_violation.extend([(batch_idx, mod_idx, int(i)) for i in bad_span])

            # ---- ③ 收样本用于直方图 ----
            # 随机下采样，免得数据量太大
            sample = flat.cpu().flatten()
            if sample.numel() > 0:
                if len(hist_buckets[mod_idx]) < max_samples:
                    take = max_samples - len(hist_buckets[mod_idx])
                    idx  = torch.randperm(sample.numel())[:take]
                    hist_buckets[mod_idx].extend(sample[idx].tolist())

        if (batch_idx + 1) % 5 == 0:
            print(f"已处理 {batch_idx + 1} 个批次 ...")

    # ========== 打印结果 ==========
    print("\n=== NaN / Inf ===")
    print("✅ 未检测到" if not nan_inf_records
          else f"⚠️  {len(nan_inf_records)} 处 NaN/Inf ，示例 {nan_inf_records[:3]}")

    print("\n=== 边界越界 (<0 或 >1) ===")
    print("✅ 全部在 [0,1]±{:.3g}".format(tol_edge) if not edge_violation
          else f"⚠️  {len(edge_violation)} 个样本越界，示例 {edge_violation[:3]}")

    print(f"\n=== 跨度过窄 (< {tol_span}) ===")
    print("✅ 全部样本跨度足够" if not span_violation
          else f"⚠️  {len(span_violation)} 个样本跨度不足，示例 {span_violation[:3]}")

    print("\n=== 模态全局数值范围 ===")
    for mod in range(5):
        mins = value_range_records[mod]["min"]
        maxs = value_range_records[mod]["max"]
        print(f"Mod{mod}: min {min(mins):.4f} ~ {max(mins):.4f} | "
              f"max {min(maxs):.4f} ~ {max(maxs):.4f}")

    # ========== 画直方图 ==========
    plt.figure(figsize=(8, 5))
    for mod in range(5):
        if hist_buckets[mod]:
            plt.hist(hist_buckets[mod],
                     bins=nbins,
                     alpha=0.5,                 # 半透明，方便重叠
                     label=f"Mod {mod}",
                     density=True,
                     histtype='stepfilled')     # 使用默认配色
    plt.title("Intensity Histogram of 5 MRI Modalities")
    plt.xlabel("Intensity (after normalisation)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.tight_layout()
    plt.show()



# =====================================================================
# 5. 测试代码
# =====================================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 初始化数据集
    dataset = KneeMRIDatasetV2MultiLabel(
        root_dir=os.environ.get("V2VA_KNEE_DICOM_ROOT", ""),
        cache_dir=os.environ.get("V2VA_KNEE_CACHE_DIR", "data/mri/nii"),
        label_xlsx=os.environ.get("V2VA_KNEE_LABEL_XLSX", "data/mri/labels.xlsx"),
        transform=get_monai_transform(),
        cache_only=True  # 仅使用缓存数据，不进行转换
    )

    # 创建数据加载器
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=custom_collate_fn
    )
    
    # 执行数据检查
    print("\n开始数据完整性检查...")
    # validate_data_integrity(loader)

    # 测试数据加载
    for views, labels in loader:
        print(f"\n模态数量: {len(views)}")
        print(f"单个模态shape: {views[0].shape}")  # 期望输出: [B, 1, D, H, W]
        print(f"标签shape: {labels.shape}")        # 期望输出: [B, 4]
        break