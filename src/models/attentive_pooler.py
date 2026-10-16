from http.client import NOT_EXTENDED
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import (
    Block,
    CrossAttention,
    CrossAttentionBlock
)
from src.utils.tensors import trunc_normal_


class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(
        self,
        num_queries=1,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        depth=4,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        topk_tokens=None,
        topk_ratio=None,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.last_query_homogeneity = None

        if topk_tokens is not None and topk_ratio is not None:
            raise ValueError("AttentivePooler: topk_tokens 与 topk_ratio 只能二选一")
        self.topk_tokens = topk_tokens
        self.topk_ratio = topk_ratio

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer)
                for i in range(depth-1)])

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _apply_topk_pooling(self, x):
        """
        对 token 维度做 top-k pooling（选择最显著 token）。
        输入: x [B, N, D]
        输出: x_topk [B, K, D]

        说明:
          - 默认关闭（topk_tokens/topk_ratio 均为 None）
          - score 使用 token 的平方范数均值（稳定且无额外参数）
        """
        if x.dim() != 3:
            return x

        bsz, num_tokens, dim = x.shape
        if num_tokens <= 1:
            return x

        if self.topk_tokens is not None:
            k = int(self.topk_tokens)
        elif self.topk_ratio is not None:
            k = int(math.ceil(num_tokens * float(self.topk_ratio)))
        else:
            return x

        k = max(1, min(k, num_tokens))
        if k >= num_tokens:
            return x

        scores = x.pow(2).mean(dim=-1)  # [B, N]
        topk_idx = scores.topk(k=k, dim=1, largest=True, sorted=False).indices  # [B, K]
        gather_idx = topk_idx.unsqueeze(-1).expand(bsz, k, dim)
        x_topk = torch.gather(x, dim=1, index=gather_idx)
        return x_topk

    @staticmethod
    def _query_homogeneity_stats(q_tokens):
        """
        评估 query 是否同质化。
        输入: q_tokens [B, Q, D]
        输出: dict，包含
          - offdiag_cos_mean: 非对角余弦相似度均值（越高越同质）
          - offdiag_cos_max: 非对角余弦相似度最大值（接近1说明塌缩）
          - offdiag_cos_std: 非对角余弦相似度标准差
        """
        if q_tokens.dim() != 3:
            return None
        bsz, q_num, _ = q_tokens.shape
        if q_num <= 1:
            return {
                "offdiag_cos_mean": 0.0,
                "offdiag_cos_max": 0.0,
                "offdiag_cos_std": 0.0,
            }

        q_norm = F.normalize(q_tokens, dim=-1)  # [B, Q, D]
        sim = torch.matmul(q_norm, q_norm.transpose(1, 2))  # [B, Q, Q]

        eye = torch.eye(q_num, device=sim.device, dtype=torch.bool).unsqueeze(0)
        offdiag = sim.masked_select(~eye).view(bsz, q_num * (q_num - 1))

        return {
            "offdiag_cos_mean": offdiag.mean().detach(),
            "offdiag_cos_max": offdiag.max().detach(),
            "offdiag_cos_std": offdiag.std(unbiased=False).detach(),
        }

    def forward(self, x):
        x = self._apply_topk_pooling(x)
        q = self.query_tokens.repeat(len(x), 1, 1) # 加入 batch 维度，形状变为 (B, q_num, embed_dim) 
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)  # self-attention on q
        self.last_query_homogeneity = self._query_homogeneity_stats(q)
        return q


class TrueCLSHeadPooler(nn.Module):
    """
    真正的 Head-CLS 聚合：
      1) prepend 一个可学习 cls token
      2) 在 [cls; x] 上做多层联合 self-attention 更新
      3) 取最终 cls 作为 pooled feature
    """
    def __init__(
        self,
        embed_dim=1024,
        num_heads=8,
        mlp_ratio=4.0,
        depth=4,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.last_query_homogeneity = None

        depth = max(1, int(depth))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=False,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.init_std = init_std
        trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks, 1):
            rescale(layer.attn.proj.weight.data, layer_id)
            rescale(layer.mlp.fc2.weight.data, layer_id)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("TrueCLSHeadPooler expects input shape [B, N, D]")

        cls = self.cls_token.repeat(len(x), 1, 1)  # [B, 1, D]
        y = torch.cat([cls, x], dim=1)  # [B, 1+N, D]
        for blk in self.blocks:
            y = blk(y)

        cls_out = y[:, :1, :]  # [B, 1, D]
        self.last_query_homogeneity = AttentivePooler._query_homogeneity_stats(cls_out)
        return cls_out


class LearnableQueryFusion(nn.Module):
    """
    对 query 维度做可学习融合:
      输入: [B, Q, D]
      输出: [B, D]
    """
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.last_attn = None
        self.last_attn_metrics = None
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q_tokens):
        # q_tokens: [B, Q, D]
        attn_logits = self.score(q_tokens).squeeze(-1)  # [B, Q]
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # [B, Q, 1]
        attn_2d = attn.squeeze(-1)
        ent = -(attn_2d * (attn_2d.clamp_min(1e-12).log())).sum(dim=1)  # [B]
        q_num = attn_2d.size(1)
        max_ent = math.log(q_num) if q_num > 1 else 1.0
        norm_ent = ent / max_ent

        self.last_attn = attn_2d.detach()
        self.last_attn_metrics = {
            "attn_entropy_mean": ent.mean().detach(),
            "attn_entropy_norm_mean": norm_ent.mean().detach(),
            "attn_max_mean": attn_2d.max(dim=1).values.mean().detach(),
        }

        fused = (q_tokens * attn).sum(dim=1)  # [B, D]
        return fused


class PlaneAgnosticCrossAttention(nn.Module):
    """
    Plane-Agnostic Cross Attention
    - 输入: A, C, S，shape 都为 [B, N, D]
    - 计算 6 个 cross attention：ac/as/ca/cs/sa/sc
    - 输出: A', C', S'，shape 保持 [B, N, D]
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.ac = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.as_ = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ca = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cs = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.sa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.sc = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def _xattn(self, layer, q, kv):
        out, _ = layer(q, kv, kv, need_weights=False)
        return out

    def forward(self, a_tokens, c_tokens, s_tokens):
        f_ac = self._xattn(self.ac, a_tokens, c_tokens)
        f_as = self._xattn(self.as_, a_tokens, s_tokens)

        f_ca = self._xattn(self.ca, c_tokens, a_tokens)
        f_cs = self._xattn(self.cs, c_tokens, s_tokens)

        f_sa = self._xattn(self.sa, s_tokens, a_tokens)
        f_sc = self._xattn(self.sc, s_tokens, c_tokens)

        a_out = a_tokens + f_ac + f_as
        c_out = c_tokens + f_ca + f_cs
        s_out = s_tokens + f_sa + f_sc
        return a_out, c_out, s_out


class PlaneGate(nn.Module):
    """
    自适应融合三个平面特征。
    输入: A', C', S'，均为 [B, N, D]
    输出: F，shape [B, D]
    """
    def __init__(
        self,
        embed_dim,
        pool_num_queries=1,
        pool_num_heads=8,
        pool_mlp_ratio=4.0,
        pool_depth=2,
        pool_dropout=0.1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        super().__init__()

        self.pool_a = AttentivePooler(
            num_queries=pool_num_queries,
            embed_dim=embed_dim,
            num_heads=pool_num_heads,
            mlp_ratio=pool_mlp_ratio,
            depth=pool_depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            topk_tokens=256,
        )
        self.pool_c = AttentivePooler(
            num_queries=pool_num_queries,
            embed_dim=embed_dim,
            num_heads=pool_num_heads,
            mlp_ratio=pool_mlp_ratio,
            depth=pool_depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            topk_tokens=256,
        )
        self.pool_s = AttentivePooler(
            num_queries=pool_num_queries,
            embed_dim=embed_dim,
            num_heads=pool_num_heads,
            mlp_ratio=pool_mlp_ratio,
            depth=pool_depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            topk_tokens=256,
        )

        self.fuse_a = LearnableQueryFusion(embed_dim=embed_dim, hidden_dim=embed_dim // 2, dropout=pool_dropout)
        self.fuse_c = LearnableQueryFusion(embed_dim=embed_dim, hidden_dim=embed_dim // 2, dropout=pool_dropout)
        self.fuse_s = LearnableQueryFusion(embed_dim=embed_dim, hidden_dim=embed_dim // 2, dropout=pool_dropout)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        self.last_weights = None

    def forward(self, a_tokens, c_tokens, s_tokens):
        # step1: independent attentive pooling -> g_A/g_C/g_S (B, D)
        g_a = self.fuse_a(self.pool_a(a_tokens))
        g_c = self.fuse_c(self.pool_c(c_tokens))
        g_s = self.fuse_s(self.pool_s(s_tokens))

        # step2: concat
        g = torch.cat([g_a, g_c, g_s], dim=-1)  # [B, 3D]

        # step3+4: 2-layer MLP + softmax
        tau = 1.5
        w = torch.softmax(self.gate(g)/tau, dim=-1)  # [B, 3]
        
        # 测试 平均权重是否合理
        # w = torch.full((g.size(0), 3), 1.0 / 3.0, device=g.device, dtype=g.dtype)  # [B, 3]

        self.last_weights = w.detach()
        w_a, w_c, w_s = w[:, 0:1], w[:, 1:2], w[:, 2:3] # 用切片,乘法会在特征维 D 上自动广播，把每个样本的单个平面权重应用到整条特征向量

        # step5: weighted sum
        fused = (w_a * g_a) + (w_c * g_c) + (w_s * g_s)  # [B, D] 每个面的embed dim各维度是否意义一致？
        return fused


class MultiHeadAttentiveClassifier(nn.Module):
    """
        多分类头分类器。

        支持三种 token 聚合方式（yaml 可控）：
            - attentive: 现有 AttentivePooler + LearnableQueryFusion
            - gap:      对 token 维做全局平均池化
            - head_cls: 真正 CLS（在头内对 [CLS; tokens] 做多层联合更新）
    """
    def __init__(
        self,
        embed_dim,
        defect_classes,
        pool_mode='Null',
        num_queries=1,
        num_heads=8,
        mlp_ratio=4.0, 
        depth=2,
        head_cls_num_heads=8,
        head_cls_mlp_ratio=4.0,
        head_cls_depth=4,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        """
        :param embed_dim: 输入特征的维度 (enc_embed_dim)
        :param defect_classes: 列表，指定每个分类头的类别数
        :param pool_mode: token 聚合方式: attentive | gap | head_cls
        :param num_queries: 每个分类头 AttentivePooler 的 query 数量
        :param num_heads: 每个分类头 AttentivePooler 的头数
        :param mlp_ratio: MLP 的隐藏层扩展比例
        :param depth: 每个 AttentivePooler 的深度
        :param head_cls_num_heads: Head-CLS 的注意力头数
        :param head_cls_mlp_ratio: Head-CLS 的 MLP ratio
        :param head_cls_depth: Head-CLS 联合 self-attn block 深度
        :param norm_layer: 归一化层类型
        :param init_std: 初始化标准差
        :param qkv_bias: 是否使用 QKV 偏置
        :param complete_block: 是否使用完整的 CrossAttentionBlock
        """
        super(MultiHeadAttentiveClassifier, self).__init__()

        self.pool_mode = str(pool_mode).lower()
        if self.pool_mode not in ('attentive', 'gap', 'head_cls', 'swin_default'):
            raise ValueError("MultiHeadAttentiveClassifier: pool_mode 必须为 'attentive' | 'gap' | 'head_cls' | 'swin_default'")

        self.poolers = None
        self.cls_poolers = None
        if self.pool_mode == 'swin_default':
            self.classification_heads = nn.ModuleList([
                nn.Linear(embed_dim, num_classes)
                for num_classes in defect_classes
            ])
        else:
            self.classification_heads = nn.ModuleList([
                StrongMLP(embed_dim=embed_dim, num_classes=num_classes, mlp_ratio=mlp_ratio, depth=2, dropout=0.1, use_residual=True)
                for num_classes in defect_classes
            ])
        self.last_plane_weights = None
        self.query_fusion = None

        if self.pool_mode == 'attentive':
            # 为每个分类头定义独立的 AttentivePooler 和 LearnableQueryFusion
            self.poolers = nn.ModuleList([
                AttentivePooler(
                    num_queries=num_queries,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    depth=depth,
                    norm_layer=norm_layer,
                    init_std=init_std,
                    qkv_bias=qkv_bias,
                    complete_block=complete_block,
                    topk_tokens=None,
                    topk_ratio=None
                )
                for _ in defect_classes
            ])
            self.query_fusion = nn.ModuleList([
                LearnableQueryFusion(embed_dim=embed_dim, hidden_dim=embed_dim // 2, dropout=0.1)
                for _ in defect_classes
            ])
        elif self.pool_mode == 'head_cls':
            # 真正 Head-CLS：每个任务一个联合 self-attn CLS 聚合器
            self.cls_poolers = nn.ModuleList([
                TrueCLSHeadPooler(
                    embed_dim=embed_dim,
                    num_heads=head_cls_num_heads,
                    mlp_ratio=head_cls_mlp_ratio,
                    depth=head_cls_depth,
                    norm_layer=norm_layer,
                    init_std=init_std,
                    qkv_bias=qkv_bias,
                )
                for _ in defect_classes
            ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.last_query_homogeneity = []
        self.last_query_fusion_metrics = []

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征，形状为 (B, N_patches, embed_dim)
        :return: logits，列表，每个元素形状为 (B, num_classes)
        """
        logits = []
        self.last_query_homogeneity = []
        self.last_query_fusion_metrics = []
        # print("ViT out:", x.min(), x.max(), torch.isfinite(x).all())
        if self.pool_mode == 'attentive':
            for pooler, fuse, head in zip(self.poolers, self.query_fusion, self.classification_heads):
                pooled = pooler(x)  # (B, q_num, embed_dim)
                pooled = fuse(pooled)  # [B, D] learnable query fusion
                self.last_query_homogeneity.append(pooler.last_query_homogeneity)
                self.last_query_fusion_metrics.append(fuse.last_attn_metrics)

                pooled = self.final_norm(pooled)
                logits.append(head(pooled))  # (B, num_classes)
            return logits

        if self.pool_mode == 'gap':
            pooled = x.mean(dim=1)  # [B, D]
            pooled = self.final_norm(pooled)
            for head in self.classification_heads:
                logits.append(head(pooled))
                self.last_query_homogeneity.append(None)
                self.last_query_fusion_metrics.append(None)
            return logits

        if self.pool_mode == 'swin_default':
            pooled = x.mean(dim=1)
            pooled = self.final_norm(pooled)
            for head in self.classification_heads:
                logits.append(head(pooled))
                self.last_query_homogeneity.append(None)
                self.last_query_fusion_metrics.append(None)
            return logits

        if self.pool_mode == 'head_cls':
            for pooler, head in zip(self.cls_poolers, self.classification_heads):
                pooled = pooler(x)[:, 0, :]  # [B, D] 取联合更新后的 CLS
                self.last_query_homogeneity.append(pooler.last_query_homogeneity)
                self.last_query_fusion_metrics.append(None)

                pooled = self.final_norm(pooled)
                logits.append(head(pooled))
            return logits


class PlaneAwareMultiHeadClassifier(nn.Module):
    """
    三平面版本：
      1) PlaneAgnosticCrossAttention
      2) PlaneGate
      3) StrongMLP 多头输出
    """
    def __init__(
        self,
        embed_dim,
        defect_classes,
        cross_num_heads=8,
        cross_dropout=0.1,
        enable_plane_cross_attention=True,
        gate_pool_num_queries=1,
        gate_pool_num_heads=8,
        gate_pool_mlp_ratio=4.0,
        gate_pool_depth=2,
        gate_pool_dropout=0.3,
        mlp_ratio=4.0,
        depth=2,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        super().__init__()

        self.enable_plane_cross_attention = enable_plane_cross_attention
        if self.enable_plane_cross_attention:
            self.plane_cross_attention = PlaneAgnosticCrossAttention(
                embed_dim=embed_dim,
                num_heads=cross_num_heads,
                dropout=cross_dropout,
            )
        else:
            self.plane_cross_attention = None

        self.plane_gate = PlaneGate(
            embed_dim=embed_dim,
            pool_num_queries=gate_pool_num_queries,
            pool_num_heads=gate_pool_num_heads,
            pool_mlp_ratio=gate_pool_mlp_ratio,
            pool_depth=gate_pool_depth,
            pool_dropout=gate_pool_dropout,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.classification_heads = nn.ModuleList([
            StrongMLP(
                embed_dim=embed_dim,
                num_classes=num_classes,
                mlp_ratio=mlp_ratio,
                depth=depth,
                dropout=0.3,
                use_residual=True,
            )
            for num_classes in defect_classes
        ])

    def forward(self, planes):
        if not isinstance(planes, (tuple, list)) or len(planes) != 3:
            raise ValueError("PlaneAwareMultiHeadClassifier expects (axial_tokens, coronal_tokens, sagittal_tokens)")

        a_tokens, c_tokens, s_tokens = planes

        if self.enable_plane_cross_attention and self.plane_cross_attention is not None:
            a_tokens, c_tokens, s_tokens = self.plane_cross_attention(a_tokens, c_tokens, s_tokens)

        fused = self.plane_gate(a_tokens, c_tokens, s_tokens)
        plane_weights = self.plane_gate.last_weights
        self.last_plane_weights = plane_weights.detach()
        fused = self.final_norm(fused)

        logits = [head(fused) for head in self.classification_heads]
        return logits


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + identity)
        # out = self.act(out)
        return out


class TokenToVolume3DClassifier(nn.Module):
    """
    将 encoder token 特征 [B, N, E] 反变换为 3D 特征图 [B, E(+3), D, H, W]，
    再通过残差 3D CNN 进行跨 token 空间聚合分类。
    """
    def __init__(
        self,
        embed_dim,
        defect_classes,
        patch_size=16,
        tubelet_size=1,
        grid_size=None,
        use_coord_channels=True,
        stem_channels=128,
        stage_channels=(128, 256, 384),
        stage_depths=(1, 1, 1),
        stage_strides=(1, 2, 2),
        dropout=0.1,
    ):
        super().__init__()
        if not (len(stage_channels) == len(stage_depths) == len(stage_strides)):
            raise ValueError("TokenToVolume3DClassifier: stage_channels/depths/strides 长度必须一致")

        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.grid_size = tuple(int(v) for v in grid_size) if grid_size is not None else None
        self.use_coord_channels = bool(use_coord_channels)

        in_channels = self.embed_dim + (3 if self.use_coord_channels else 0)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.GELU(),
        )

        self.stages = nn.ModuleList()
        cur_ch = stem_channels

        # stages abalation 1 comment
        for out_ch, depth, stride in zip(stage_channels, stage_depths, stage_strides):
            blocks = []
            for i in range(int(depth)):
                blk_stride = int(stride) if i == 0 else 1
                blk_in = cur_ch if i == 0 else int(out_ch)
                blocks.append(ResBlock3D(blk_in, int(out_ch), stride=blk_stride))
            self.stages.append(nn.Sequential(*blocks))
            cur_ch = int(out_ch)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classification_heads = nn.ModuleList([
            nn.Linear(cur_ch, num_classes) for num_classes in defect_classes
        ])
        self.final_channels = cur_ch  # 供 Concat + Unified Linear 模式使用

        # 兼容现有训练日志逻辑
        self.last_query_homogeneity = []
        self.last_query_fusion_metrics = []
        self.last_plane_weights = None

    @staticmethod
    def _build_coords(batch_size, d, h, w, device, dtype):
        zz = torch.linspace(-1.0, 1.0, steps=d, device=device, dtype=dtype)
        yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        z_grid, y_grid, x_grid = torch.meshgrid(zz, yy, xx, indexing='ij')
        coords = torch.stack([z_grid, y_grid, x_grid], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        return coords

    @staticmethod
    def _factor_triplets(n):
        triplets = []
        for d in range(1, int(round(n ** (1.0 / 3.0))) + 2):
            if n % d != 0:
                continue
            n1 = n // d
            for h in range(1, int(math.sqrt(n1)) + 1):
                if n1 % h != 0:
                    continue
                w = n1 // h
                triplets.append((d, h, w))
        if not triplets:
            triplets = [(1, 1, n)]
        return triplets

    def _resolve_grid(self, num_tokens, volume_shape=None):
        if self.grid_size is not None:
            if self.grid_size[0] * self.grid_size[1] * self.grid_size[2] != num_tokens:
                raise ValueError(
                    f"TokenToVolume3DClassifier: 指定 grid_size={self.grid_size} 与 token 数 N={num_tokens} 不匹配"
                )
            return self.grid_size

        hint = None
        if volume_shape is not None:
            if len(volume_shape) != 3:
                raise ValueError("TokenToVolume3DClassifier: volume_shape 必须是 (D, H, W)")
            d, h, w = [int(v) for v in volume_shape]
            hint = (
                max(1, d // self.tubelet_size),
                max(1, h // self.patch_size),
                max(1, w // self.patch_size),
            )
            if hint[0] * hint[1] * hint[2] == num_tokens:
                return hint

        candidates = self._factor_triplets(int(num_tokens))

        def score(tri):
            d, h, w = tri
            s = abs(d - h) + abs(h - w) + abs(d - w)
            if hint is not None:
                s = s + 0.5 * (abs(d - hint[0]) + abs(h - hint[1]) + abs(w - hint[2]))
            return s

        return min(candidates, key=score)

    def forward_features(self, x, volume_shape=None):
        """返回 3D CNN backbone 提取的全局特征 [B, final_channels]，不经过分类头 Linear。"""
        if x.dim() != 3:
            raise ValueError("TokenToVolume3DClassifier expects input shape [B, N, E]")

        bsz, num_tokens, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"TokenToVolume3DClassifier: 输入 embed_dim={embed_dim} 与初始化 embed_dim={self.embed_dim} 不一致"
            )

        d, h, w = self._resolve_grid(num_tokens=num_tokens, volume_shape=volume_shape)
        if d * h * w != num_tokens:
            raise ValueError(
                f"TokenToVolume3DClassifier: 无法将 N={num_tokens} reshape 到 (D,H,W)=({d},{h},{w})"
            )

        vol = x.transpose(1, 2).contiguous().view(bsz, embed_dim, d, h, w)
        if self.use_coord_channels:
            coords = self._build_coords(bsz, d, h, w, x.device, x.dtype)
            vol = torch.cat([vol, coords], dim=1)

        y = self.stem(vol)

        # stages abalation 2 comment
        for stage in self.stages:
            y = stage(y)

        y = self.pool(y).flatten(1)
        y = self.dropout(y)
        return y

    def forward(self, x, volume_shape=None):
        y = self.forward_features(x, volume_shape=volume_shape)
        logits = [head(y) for head in self.classification_heads]
        return logits

    # 原版适配medmnist的forward
    # def forward(self, x, volume_shape=None):
    #     if x.dim() != 3:
    #         raise ValueError("TokenToVolume3DClassifier expects input shape [B, N, E]")

    #     bsz, num_tokens, embed_dim = x.shape
    #     if embed_dim != self.embed_dim:
    #         raise ValueError(
    #             f"TokenToVolume3DClassifier: 输入 embed_dim={embed_dim} 与初始化 embed_dim={self.embed_dim} 不一致"
    #         )

    #     d, h, w = self._resolve_grid(num_tokens=num_tokens, volume_shape=volume_shape)
    #     if d * h * w != num_tokens:
    #         raise ValueError(
    #             f"TokenToVolume3DClassifier: 无法将 N={num_tokens} reshape 到 (D,H,W)=({d},{h},{w})"
    #         )

    #     vol = x.transpose(1, 2).contiguous().view(bsz, embed_dim, d, h, w)
    #     if self.use_coord_channels:
    #         coords = self._build_coords(bsz, d, h, w, x.device, x.dtype)
    #         vol = torch.cat([vol, coords], dim=1)

    #     y = self.stem(vol)

    #     # stages abalation 2 comment
    #     for stage in self.stages:
    #         y = stage(y)

    #     y = self.pool(y).flatten(1)
    #     y = self.dropout(y)

    #     logits = [head(y) for head in self.classification_heads]
    #     return logits
    
class StrongMLP(nn.Module):
    def __init__(self, embed_dim, num_classes, mlp_ratio=4, depth=2, dropout=0.3, use_residual=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        depth = max(1, int(depth))

        def make_ffn_block():
            layers = [
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, embed_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        # 采用更轻量、稳定的 FFN 堆叠，替代原先两套超宽 MLP
        self.blocks = nn.ModuleList([make_ffn_block() for _ in range(depth)])
        self.block_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.use_residual = use_residual
        self.post_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, num_classes) #  权重形状：weight.shape = (num_classes, embed_dim), weight[i] 就是W_i也就是CosFace论文里的类中心向量。

    def forward(self, x):
        x = self.layer_norm(x)
        for blk, norm in zip(self.blocks, self.block_norms):
            h = blk(norm(x))
            x = (x + h) if self.use_residual else h
        x = self.post_norm(x)
        return self.out(x)


class MRNetTriplePlaneClassifier(nn.Module):
    """MRNet three-plane classifier.

    Three independent TokenToVolume3DClassifiers, one per plane (axial/coronal/sagittal).
    Each plane's encoder tokens are fed into its own classifier head.
    Final logits = average of three heads.
    """

    def __init__(
        self,
        embed_dim,
        defect_classes,
        patch_size=16,
        tubelet_size=1,
        grid_size=None,
        use_coord_channels=True,
        stem_channels=128,
        stage_channels=(128, 256, 384),
        stage_depths=(1, 1, 1),
        stage_strides=(1, 2, 2),
        dropout=0.1,
        plane_names=("axial", "coronal", "sagittal"),
    ):
        super().__init__()
        self.plane_names = plane_names
        self.heads = nn.ModuleDict()
        for name in plane_names:
            self.heads[name] = TokenToVolume3DClassifier(
                embed_dim=embed_dim,
                defect_classes=defect_classes,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                grid_size=grid_size,
                use_coord_channels=use_coord_channels,
                stem_channels=stem_channels,
                stage_channels=stage_channels,
                stage_depths=stage_depths,
                stage_strides=stage_strides,
                dropout=dropout,
            )
        # Compat with existing logging
        self.last_query_homogeneity = []
        self.last_query_fusion_metrics = []
        self.last_plane_weights = None

    def forward(self, tokens_dict, volume_shape=None):
        """tokens_dict: {'axial': [B, N, E], 'coronal': [B, N, E], 'sagittal': [B, N, E]}"""
        logits_list = []
        for name in self.plane_names:
            tokens = tokens_dict[name]
            logits = self.heads[name](tokens, volume_shape=volume_shape)
            logits_list.append(logits)

        # Average logits across planes
        num_tasks = len(logits_list[0])
        averaged = []
        for t in range(num_tasks):
            avg = torch.stack([l[t] for l in logits_list], dim=0).mean(dim=0)
            averaged.append(avg)
        return averaged


class MRNetTriplePlanePerTaskClassifier(nn.Module):
    """MRNet per-task per-plane classifier.

    3 views × N defects = 3*N independent TokenToVolume3DClassifiers.
    Each has its own complete 3D CNN backbone + single linear head.
    Final logits = average of 3 plane logits per defect.
    """

    def __init__(
        self,
        embed_dim,
        defect_classes,
        patch_size=16,
        tubelet_size=1,
        grid_size=None,
        use_coord_channels=True,
        stem_channels=128,
        stage_channels=(128, 256, 384),
        stage_depths=(1, 1, 1),
        stage_strides=(1, 2, 2),
        dropout=0.1,
        plane_names=("axial", "coronal", "sagittal"),
        defect_names=None,
        use_feature_concat=False,
    ):
        super().__init__()
        self.plane_names = plane_names
        self.defect_names = defect_names
        self.num_tasks = len(defect_classes)
        self.use_feature_concat = use_feature_concat

        # 3 views × N defects = 3*N independent classifiers
        self.heads = nn.ModuleDict()
        for pname in plane_names:
            for didx, ncls in enumerate(defect_classes):
                key = f"{pname}_{defect_names[didx]}"
                self.heads[key] = TokenToVolume3DClassifier(
                    embed_dim=embed_dim,
                    defect_classes=[ncls],  # single linear head for this task
                    patch_size=patch_size,
                    tubelet_size=tubelet_size,
                    grid_size=grid_size,
                    use_coord_channels=use_coord_channels,
                    stem_channels=stem_channels,
                    stage_channels=stage_channels,
                    stage_depths=stage_depths,
                    stage_strides=stage_strides,
                    dropout=dropout,
                )

        if use_feature_concat:
            # 获取 backbone 输出通道数（取第一个 head 的 final_channels）
            first_key = f"{plane_names[0]}_{defect_names[0]}"
            final_ch = self.heads[first_key].final_channels
            # 每个任务一个 Unified Linear(3*final_ch, ncls)
            self.unified_heads = nn.ModuleDict()
            for didx, ncls in enumerate(defect_classes):
                dname = defect_names[didx]
                self.unified_heads[dname] = nn.Linear(len(plane_names) * final_ch, ncls)

        # Compat with existing logging
        self.last_query_homogeneity = []
        self.last_query_fusion_metrics = []
        self.last_plane_weights = None

    def forward(self, tokens_dict, volume_shape=None):
        """tokens_dict: {'axial': [B, N, E], 'coronal': [B, N, E], 'sagittal': [B, N, E]}"""
        if self.use_feature_concat:
            # Concat + Unified Linear 模式
            outputs = []
            for didx in range(self.num_tasks):
                dname = self.defect_names[didx]
                plane_feats = []
                for pname in self.plane_names:
                    key = f"{pname}_{dname}"
                    tokens = tokens_dict[pname]
                    # forward_features 返回 [B, final_ch]，不经过 Linear 头
                    feat = self.heads[key].forward_features(tokens, volume_shape=volume_shape)
                    plane_feats.append(feat)
                # 三平面特征 concat → [B, 3*final_ch] → Unified Linear → logit
                concat_feat = torch.cat(plane_feats, dim=-1)
                logit = self.unified_heads[dname](concat_feat)
                outputs.append(logit)
            return outputs
        else:
            # 原始行为：三平面 logits 平均
            outputs = []
            for didx in range(self.num_tasks):
                dname = self.defect_names[didx]
                plane_logits = []
                for pname in self.plane_names:
                    key = f"{pname}_{dname}"
                    tokens = tokens_dict[pname]
                    # Single-task classifier returns a list of 1 logit
                    logit = self.heads[key](tokens, volume_shape=volume_shape)[0]
                    plane_logits.append(logit)
                # Average across 3 planes for this task
                avg = torch.stack(plane_logits, dim=0).mean(dim=0)
                outputs.append(avg)
                # Max across 3 planes for this task
                # max_val = torch.stack(plane_logits, dim=0).max(dim=0).values
                # outputs.append(max_val)
            return outputs


# 测试脚本
if __name__ == "__main__":
    B = 4  # batch size
    N_patches = 8000
    embed_dim = 1024
    x = torch.randn(B, N_patches, embed_dim)

    defect_classes = [4, 4, 4, 3, 3]  # 五个分类头，类别数分别为 4, 4, 4, 3, 3
    model = MultiHeadAttentiveClassifier(
        embed_dim=embed_dim,
        defect_classes=defect_classes,
        pool_mode='attentive',
        num_heads=8,
        mlp_ratio=4.0,
        depth=2,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    )

    logits = model(x)
    for i, logit in enumerate(logits):
        print(f"Logits for classification head {i}: shape {logit.shape}")
        print(logit)

# class StrongMLP(nn.Module):
#     def __init__(self, embed_dim, num_classes, mlp_ratio=4):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(embed_dim)
#         hidden = int(embed_dim * mlp_ratio)
#         self.fc1 = nn.Linear(embed_dim, hidden)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden, embed_dim)
#         self.norm = nn.LayerNorm(embed_dim)
#         self.out = nn.Linear(embed_dim, num_classes)
#     def forward(self, x):
#         x = self.layer_norm(x)
#         # print("layer_norm:", x.min(), x.max(), torch.isnan(x).any())
#         h = self.fc2(self.act(self.fc1(x)))
#         # print("mlp:", h.min(), h.max(), torch.isnan(h).any())
#         # x = self.norm(x + h)     # 残差增强表达能力 最后logits不加残差
#         # print("norm:", x.min(), x.max(), torch.isnan(x).any())
#         return self.out(h)
    

