import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
from torch.utils.data import DataLoader, random_split

from app.vjepa.transforms_mri import MakeMRITransforms


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
        # "pdcor",   # 模态1：PD加权冠状位
        "pdsag",  # 模态2：PD Dixon矢状位
        # "pdtra",   # 模态3：PD加权横断位
        "t1sag",     # 模态4：T1加权矢状位
        "t2sag",       # 模态5：T2加权矢状位
    ]

    def __init__(
        self,
        training: bool,
        root_dir: Union[str, Path],
        label_xlsx: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        transform=None,
        defect:list = [],
        cache_only: bool = True,
    ) -> None:
        super().__init__()

        self.training   = training
        self.root_dir   = Path(root_dir).expanduser().resolve()
        self.cache_dir  = (Path(cache_dir).expanduser().resolve()
                           if cache_dir else self.root_dir.with_name(self.root_dir.name + "_nifti_cache"))
        self.transform  = transform
        self.defect     = defect
        self.cache_only = cache_only

        # ------------------------------------------------------------
        # ① 读取标签
        # ------------------------------------------------------------
        df = pd.read_excel(label_xlsx).dropna()
        df[defect] = df[defect].astype(int)
        self.label_map = {  # 测试取标签
            str(r["Patient ID"]).strip(): r[defect].tolist()
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
        
        seed = np.random.randint(0, 1e6)

        data = []
        for pref in self.MODALITY_PREFIXES:
            # 从缓存加载NIfTI文件
            nifti_path = self.cache_dir / pid / f"{pref}.nii.gz"
            try:
                img_np = nib.load(nifti_path).get_fdata()
            except FileNotFoundError:
                raise RuntimeError(f"缓存缺失: {nifti_path}")  # 理论上不会发生

            # Ensure data is 3D and has depth of 20
            # if img_np.shape[-1] != 20:  # 深度维在最后
            #     print(f"Invalid depth for file {pid}{pref}: {img_np.shape[-1]}")
            # Apply transformations before stacking
            if self.transform:  
                # print("current seed:", seed)
                use_transforms = self.transform(training=self.training)       # 实例化
                img_tensor = use_transforms(img_np, seed)   # 同一患者不同模态传递同一种子，确保各序列变换后仍配准

            # save transformed niigz for checking
            # volume_save = img_tensor.numpy().astype(np.float32)
            # debug_dir = os.environ.get("V2VA_NII_DEBUG_DIR", "outputs/nii_debug")
            # nib.save(nib.Nifti1Image(volume_save, np.eye(4)),
            #          f"{debug_dir}/{idx}{pref}_transformed.nii.gz")
            # print(f"Saved transformed nifti for {pref}")
            data.append(img_tensor)

        data = torch.stack(data, dim=0)  # Shape: (3, D, H, W)
        # print("__getitem__ data:", data.shape) 

        return data, label
    
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
def get_monai_transform(spatial_size=(20, 320, 320)):
    """训练集数据增强管道（包含空间变换和强度归一化）"""
    return Compose([
        # 基于百分位的强度归一化（排除0.5%~99.5%外的异常值）
        ScaleIntensityRangePercentilesd(
            keys="image", 
            lower=1, 
            upper=100, 
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

def get_monai_val_transform(spatial_size=(20, 320, 320)):
    """验证集变换管道（仅归一化和尺寸调整）"""
    return Compose([
        ScaleIntensityRangePercentilesd(
            keys="image", 
            lower=1, 
            upper=100, 
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
    """
    自定义批处理函数：合并多模态 MRI 数据和标签
    :param batch: [(data, label), ...]，其中 data 的形状为 [3, 20, 320, 320]，label 的形状为 [num_labels]
    :return: 
        - batched_data: [B, 3, 20, 320, 320]，批量化的 MRI 数据
        - batched_labels: [B, num_labels]，批量化的标签
    """
    # 解压 batch，分别获取数据和标签
    data, labels = zip(*batch)  # data: tuple([3, 20, 320, 320]), labels: tuple([num_labels])

    # 将数据堆叠到 batch 维度
    batched_data = torch.stack(data, dim=0)  # [B, 3, 20, 320, 320]
    print(f"Batched data shape: {batched_data.shape}")
    # 将标签堆叠到 batch 维度
    batched_labels = torch.stack(labels, dim=0)  # [B, num_labels]
    print(f"Batched labels shape: {batched_labels.shape}")
    return batched_data, batched_labels

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



def make_dataloader(training, dataset_path, label_path, transform, batch_size, defect, num_workers=8, val_split=0.2):
    """
    创建训练和验证的 DataLoader
    :param training: 是否为训练模式
    :param dataset_path: 数据集路径
    :param label_path: 标签文件路径
    :param transform: 数据增强或预处理方法
    :param batch_size: 每个批次的样本数
    :param num_workers: DataLoader 的工作线程数
    :param val_split: 验证集划分比例
    :return: train_loader 或 val_loader
    """
    # 加载完整数据集
    dataset = KneeMRIDatasetV2MultiLabel(
        training=training,
        root_dir=dataset_path,
        cache_dir=dataset_path,
        label_xlsx=label_path,
        transform=transform,
        defect=defect,
        cache_only=True
    )

    # 划分训练集和验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 根据模式返回对应的 DataLoader
    if training:
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # 训练集需要打乱
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证集不需要打乱
            num_workers=num_workers,
            pin_memory=True
        )
    

# =====================================================================
# 5. 测试代码
# =====================================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 初始化数据集
    dataset = KneeMRIDatasetV2MultiLabel(
        training=True,
        root_dir=os.environ.get("V2VA_KNEE_DICOM_ROOT", "data/mri/dicom"),
        cache_dir=os.environ.get("V2VA_KNEE_CACHE_DIR", "data/mri/nii"),
        label_xlsx=os.environ.get("V2VA_KNEE_LABEL_XLSX", "data/mri/labels.xlsx"),
        transform=MakeMRITransforms,
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
    print(f"数据集样本数: {len(dataset)}")
    # 执行数据检查
    # print("\n开始数据完整性检查...")
    # validate_data_integrity(loader)

    # 测试数据加载
    for data, labels in loader:
        print(f"模态shape: {data.shape}")  # 期望输出: [B, 3, D, H, W]
        print(f"标签shape: {labels.shape}")        # 期望输出: [B, 5]
        break