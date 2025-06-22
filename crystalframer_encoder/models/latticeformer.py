"""
簡略化されたLatticeformerモデル
CrystalFramerから必要な部分のみを抽出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
import copy

from .pooling import max_pool, avr_pool


class LatticeformerParams:
    """Latticeformerのパラメータクラス"""
    
    def __init__(self, **kwargs):
        # デフォルト値
        self.domain = "real"
        self.lattice_range = 2
        self.minimum_range = True
        self.adaptive_cutoff_sigma = -3.5
        self.gauss_lb_real = 0.5
        self.gauss_lb_reci = 0.5
        self.scale_real = [1.4]
        self.scale_reci = [2.2]
        self.normalize_gauss = True
        self.value_pe_dist_real = 64
        self.value_pe_dist_coef = 1.0
        self.value_pe_dist_max = -10.0
        self.value_pe_dist_wscale = 1.0
        self.value_pe_wave_real = 0
        self.value_pe_dist_reci = 0
        self.value_pe_wave_reci = 0
        self.value_pe_angle_real = 16
        self.value_pe_angle_coef = 1.0
        self.value_pe_angle_wscale = 4.0
        self.positive_func_beta = 0.1
        self.layer_index = -1
        self.gauss_state = "q"
        self.frame_method = "max"
        self.frame_mode = "both"
        self.cos_abs = 1
        self.symm_break_noise = 1e-5
        
        # 引数で上書き
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def parseFromArgs(self, args):
        """引数オブジェクトからパラメータを設定"""
        for key in self.__dict__:
            if hasattr(args, key):
                setattr(self, key, getattr(args, key))


class SimplifiedLatticeMultiheadAttention(nn.Module):
    """
    オリジナルのIndexedLatticeMultiheadAttentionと互換性のあるパラメータ構造を持つ
    簡略化されたLattice Multihead Attention
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, kdim=0, vdim=0, params=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # kdim, vdimの処理（オリジナルと同じ）
        self.kdim = kdim * num_heads if kdim is not None and kdim > 0 else embed_dim
        self.vdim = vdim * num_heads if vdim is not None and vdim > 0 else embed_dim
        
        # オリジナルと同じパラメータ名を使用
        self.gauss_scale = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.atten_scale = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        # 分離されたQ,K,V投影重み（オリジナルと同じ構造）
        self.q_proj_weight = nn.Parameter(torch.empty(self.kdim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(self.kdim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.empty(self.vdim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(self.kdim * 2 + self.vdim))
        
        # 出力投影
        self.out_proj = nn.Linear(self.vdim, embed_dim)
        
        # CrystalFramer特有のパラメータ
        if params is None:
            # デフォルトパラメータ
            self.lattice_pos_weights = nn.Parameter(torch.empty(self.kdim))
            self.pe_dist_proj = nn.Parameter(torch.empty(1, num_heads, self.head_dim, 64))
            self.proj_angle1 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, 16))
            self.proj_angle2 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, 16))
            self.proj_angle3 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, 16))
        else:
            # パラメータに基づいて設定
            if params.gauss_state.startswith("q"):
                self.lattice_pos_weights = nn.Parameter(torch.empty(self.kdim))
            elif params.gauss_state == "1":
                self.lattice_pos_weights = nn.Parameter(torch.empty(self.kdim))
            elif params.gauss_state.startswith("x"):
                self.lattice_pos_weights = nn.Parameter(torch.empty(num_heads, embed_dim))
            else:
                self.lattice_pos_weights = nn.Parameter(torch.empty(self.kdim))
            
            # 距離投影パラメータ
            if hasattr(params, 'value_pe_dist_real') and params.value_pe_dist_real > 0:
                self.pe_dist_proj = nn.Parameter(torch.empty(1, num_heads, self.head_dim, params.value_pe_dist_real))
            else:
                self.pe_dist_proj = nn.Parameter(torch.empty(1, num_heads, self.head_dim, 64))
            
            # 角度投影パラメータ
            if hasattr(params, 'value_pe_angle_real') and params.value_pe_angle_real > 0:
                angle_dim = params.value_pe_angle_real
            else:
                angle_dim = 16
            
            self.proj_angle1 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, angle_dim))
            self.proj_angle2 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, angle_dim))
            self.proj_angle3 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, angle_dim))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # オリジナルと同じ初期化
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        
        # lattice_pos_weightsの初期化
        nn.init.normal_(self.lattice_pos_weights, 0., (self.kdim // self.num_heads) ** -0.5)
        
        # pe_dist_projの初期化
        W = self.pe_dist_proj.view(-1, self.pe_dist_proj.shape[-1])
        nn.init.xavier_uniform_(W, (W.shape[-1]) ** -0.5)
        
        # 角度投影の初期化
        for proj in [self.proj_angle1, self.proj_angle2, self.proj_angle3]:
            W = proj.view(-1, proj.shape[-1])
            nn.init.xavier_uniform_(W, (W.shape[-1]) ** -0.5)
    
    def forward(self, x, attn_mask=None):
        """
        簡略化されたforward処理
        実際のCrystalFramerの複雑な処理は省略し、標準的なattentionを実行
        """
        B, N, C = x.shape
        
        # Q, K, V投影（オリジナルと同じ分離された重み）
        q = F.linear(x, self.q_proj_weight, self.in_proj_bias[:self.kdim])
        k = F.linear(x, self.k_proj_weight, self.in_proj_bias[self.kdim:self.kdim*2])
        v = F.linear(x, self.v_proj_weight, self.in_proj_bias[self.kdim*2:])
        
        # Multi-head形状に変換
        head_dim_q = self.kdim // self.num_heads
        head_dim_k = self.kdim // self.num_heads
        head_dim_v = self.vdim // self.num_heads
        
        q = q.view(B, N, self.num_heads, head_dim_q).transpose(1, 2)
        k = k.view(B, N, self.num_heads, head_dim_k).transpose(1, 2)
        v = v.view(B, N, self.num_heads, head_dim_v).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = head_dim_q ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn = attn + attn_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.vdim)
        
        # 出力投影
        out = self.out_proj(out)
        
        return out


class SimplifiedLatticeformerEncoderLayer(nn.Module):
    """
    オリジナルのIndexedLatticeformerEncoderLayerと互換性のあるパラメータ構造を持つ
    簡略化されたLatticeformerエンコーダレイヤー
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", kdim=0, vdim=0, params=None, no_layer_norm=False):
        super().__init__()
        
        # Self-attention（オリジナルと同じパラメータ構造）
        self.self_attn = SimplifiedLatticeMultiheadAttention(
            d_model, nhead, dropout, kdim, vdim, params
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization（オリジナルと同じ条件分岐）
        # no_layer_norm=Trueの場合、パラメータを持たない恒等関数を使用
        self.norm1 = nn.LayerNorm(d_model) if not no_layer_norm else (lambda x: x)
        self.norm2 = nn.LayerNorm(d_model) if not no_layer_norm else (lambda x: x)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x):
        """
        簡略化されたforward処理
        """
        # Self-attention
        x2 = self.self_attn(x)
        x = self.norm1(x + self.dropout1(x2))
        
        # Feed-forward
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        
        return x


class SimplifiedLatticeformerEncoder(nn.Module):
    """
    オリジナルのIndexedLatticeformerEncoderと互換性のあるパラメータ構造を持つ
    簡略化されたLatticeformerエンコーダ
    """
    
    def __init__(self, 
                 model_dim: int,
                 head_num: int,
                 num_encoder_layers: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 lattice_params=None,
                 k_dim: int = 0,
                 v_dim: int = 0,
                 t_fixup_init: bool = False,
                 **kwargs):
        super().__init__()
        
        self.model_dim = model_dim
        self.head_num = head_num
        self.num_layers = num_encoder_layers
        
        # オリジナルと同じ条件分岐: t_fixup_init=True の場合 no_layer_norm=True
        no_layer_norm = t_fixup_init
        
        # レイヤーをModuleListで管理（オリジナルと同じ構造）
        self.layers = nn.ModuleList([
            SimplifiedLatticeformerEncoderLayer(
                d_model=model_dim,
                nhead=head_num,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                kdim=k_dim,
                vdim=v_dim,
                params=lattice_params,
                no_layer_norm=no_layer_norm
            )
            for _ in range(num_encoder_layers)
        ])
        
        # オリジナルにはpos_encodingは存在しない（削除）
        
    def forward(self, x, pos, batch, trans_vec, sizes, this_epoch=None):
        """
        Args:
            x: 原子特徴量 [total_atoms, model_dim]
            pos: 原子座標 [total_atoms, 3]
            batch: バッチインデックス [total_atoms]
            trans_vec: 格子ベクトル [batch_size, 3, 3]
            sizes: 各構造の原子数 [batch_size]
        
        Returns:
            エンコードされた特徴量 [total_atoms, model_dim]
        """
        # オリジナルでは位置エンコーディングは複雑な格子計算で処理される
        # ここでは簡略化のため、入力をそのまま使用
        
        # バッチごとに処理（簡略化）
        batch_size = sizes.shape[0]
        outputs = []
        
        start_idx = 0
        for i in range(batch_size):
            end_idx = start_idx + sizes[i].item()
            
            # 単一構造の特徴量を取得
            x_single = x[start_idx:end_idx].unsqueeze(0)  # [1, num_atoms, model_dim]
            
            # 各レイヤーを順次適用
            for layer in self.layers:
                x_single = layer(x_single)  # [1, num_atoms, model_dim]
            
            outputs.append(x_single.squeeze(0))  # [num_atoms, model_dim]
            start_idx = end_idx
        
        return torch.cat(outputs, dim=0)


class Latticeformer(nn.Module):
    """
    簡略化されたLatticeformerモデル
    """
    
    def __init__(self, params):
        super().__init__()
        
        self.params = params
        
        # パラメータを取得
        model_dim = getattr(params, 'model_dim', 128)
        num_layers = getattr(params, 'num_layers', 4)
        head_num = getattr(params, 'head_num', 8)
        ff_dim = getattr(params, 'ff_dim', 512)
        dropout = getattr(params, 'dropout', 0.1)
        t_activation = getattr(params, 't_activation', 'relu')
        embedding_dim = copy.deepcopy(getattr(params, 'embedding_dim', [128]))
        pooling = getattr(params, 'pooling', "max")
        pre_pooling_op = getattr(params, 'pre_pooling_op', "no")
        norm_type = getattr(params, 'norm_type', "no")
        t_fixup_init = getattr(params, 't_fixup_init', True)
        
        self.pooling = pooling
        
        # 原子特徴量の次元
        self.ATOM_FEAT_DIM = 98
        
        # 入力埋め込み層
        self.input_embeddings = nn.Linear(self.ATOM_FEAT_DIM, model_dim, bias=False)
        emb_scale = model_dim**(-0.5)
        if t_fixup_init:
            emb_scale *= (9*num_layers)**(-1/4)
        nn.init.normal_(self.input_embeddings.weight, mean=0, std=emb_scale)
        
        # LatticeformerParamsオブジェクトを作成
        lattice_params = LatticeformerParams()
        lattice_params.parseFromArgs(params)
        
        # エンコーダ
        self.encoder = SimplifiedLatticeformerEncoder(
            model_dim=model_dim,
            head_num=head_num,
            num_encoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=t_activation,
            lattice_params=lattice_params,
            k_dim=getattr(params, 'k_dim', 0),
            v_dim=getattr(params, 'v_dim', 0),
            t_fixup_init=t_fixup_init
        )
        
        # 正規化層の設定
        if norm_type == "bn":
            norm_type = nn.BatchNorm1d
        elif norm_type == "ln":
            norm_type = nn.LayerNorm
        elif norm_type == "in":
            norm_type = nn.InstanceNorm1d
        elif norm_type in ["id", "no"]:
            norm_type = nn.Identity
        else:
            raise NotImplementedError(f"norm_type: {norm_type}")
        
        # プーリング前の投影層
        dim_pooled = model_dim
        self.proj_before_pooling = lambda x: x
        
        if pre_pooling_op == "w+bn+relu":
            dim_pooled = embedding_dim.pop(0)
            self.proj_before_pooling = nn.Sequential(
                nn.Linear(model_dim, dim_pooled),
                norm_type(dim_pooled),
                nn.ReLU(True)
            )
        elif pre_pooling_op == "w+relu":
            dim_pooled = embedding_dim.pop(0)
            self.proj_before_pooling = nn.Sequential(
                nn.Linear(model_dim, dim_pooled),
                nn.ReLU(True)
            )
        elif pre_pooling_op == "relu":
            self.proj_before_pooling = nn.ReLU(True)
        elif pre_pooling_op == "no":
            pass
        else:
            raise NotImplementedError(f"pre_pooling_op: {pre_pooling_op}")
        
        # プーリング層
        if self.pooling == "max":
            self.pooling_layer = max_pool
        elif self.pooling == "avr":
            self.pooling_layer = avr_pool
        else:
            raise NotImplementedError(f"pooling: {pooling}")
        
        # 最終MLP（回帰用だが、エンコーダでは使用しない）
        final_dim = 1 if isinstance(getattr(params, 'targets', 'dummy'), str) else 1
        
        in_dim = [dim_pooled] + embedding_dim[:-1] if embedding_dim else [dim_pooled]
        out_dim = embedding_dim if embedding_dim else [dim_pooled]
        
        layers = []
        for di, do in zip(in_dim, out_dim):
            layers.append(nn.Linear(di, do))
            layers.append(norm_type(do))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(out_dim[-1] if out_dim else dim_pooled, final_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, data, this_epoch=None):
        """
        Args:
            data: torch_geometric.data.Data または Batch
            this_epoch: エポック数（互換性のため）
        
        Returns:
            回帰出力（エンコーダでは使用しない）
        """
        x = data.x
        pos = data.pos
        batch = data.batch
        trans = data.trans_vec
        sizes = data.sizes
        
        # 入力埋め込み
        x = self.input_embeddings(x)
        
        # エンコーダ処理
        x = self.encoder(x, pos, batch, trans, sizes, this_epoch)
        
        # プーリング前の投影
        x = self.proj_before_pooling(x)
        
        # プーリング
        if self.pooling.startswith("pma"):
            x = self.pooling_layer(x, batch, sizes.shape[0])
        else:
            x = self.pooling_layer(x, batch, sizes)
        
        # 最終MLP（エンコーダでは通常使用しない）
        output = self.mlp(x)
        
        return output
