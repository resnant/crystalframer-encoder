"""
プーリング機能
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def max_pool(x, batch, sizes):
    """
    最大プーリング
    
    Args:
        x: 特徴量 [total_atoms, feature_dim]
        batch: バッチインデックス [total_atoms]
        sizes: 各構造の原子数 [batch_size]
    
    Returns:
        プーリングされた特徴量 [batch_size, feature_dim]
    """
    return scatter(x, batch, dim=0, reduce='max')


def avr_pool(x, batch, sizes):
    """
    平均プーリング
    
    Args:
        x: 特徴量 [total_atoms, feature_dim]
        batch: バッチインデックス [total_atoms]
        sizes: 各構造の原子数 [batch_size]
    
    Returns:
        プーリングされた特徴量 [batch_size, feature_dim]
    """
    return scatter(x, batch, dim=0, reduce='mean')


def sum_pool(x, batch, sizes):
    """
    合計プーリング
    
    Args:
        x: 特徴量 [total_atoms, feature_dim]
        batch: バッチインデックス [total_atoms]
        sizes: 各構造の原子数 [batch_size]
    
    Returns:
        プーリングされた特徴量 [batch_size, feature_dim]
    """
    return scatter(x, batch, dim=0, reduce='add')


class AttentionPooling(nn.Module):
    """
    注意機構ベースのプーリング
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = feature_dim
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, batch, sizes):
        """
        Args:
            x: 特徴量 [total_atoms, feature_dim]
            batch: バッチインデックス [total_atoms]
            sizes: 各構造の原子数 [batch_size]
        
        Returns:
            プーリングされた特徴量 [batch_size, feature_dim]
        """
        # 注意重みを計算
        attention_weights = self.attention(x)  # [total_atoms, 1]
        
        # バッチごとにソフトマックスを適用
        attention_weights = scatter_softmax(attention_weights.squeeze(-1), batch, dim=0)
        
        # 重み付き平均を計算
        weighted_x = x * attention_weights.unsqueeze(-1)
        return scatter(weighted_x, batch, dim=0, reduce='add')


def scatter_softmax(src, index, dim=0):
    """
    バッチごとのソフトマックス
    """
    # 各バッチの最大値を取得
    max_values = scatter(src, index, dim=dim, reduce='max')
    max_values = max_values[index]
    
    # 数値安定性のために最大値を引く
    src_shifted = src - max_values
    
    # 指数関数を適用
    exp_src = torch.exp(src_shifted)
    
    # 各バッチの合計を計算
    sum_exp = scatter(exp_src, index, dim=dim, reduce='add')
    sum_exp = sum_exp[index]
    
    # ソフトマックスを計算
    return exp_src / (sum_exp + 1e-8)


class SetToSetPooling(nn.Module):
    """
    Set2Set プーリング
    """
    
    def __init__(self, feature_dim: int, num_steps: int = 3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        
        # LSTM
        self.lstm = nn.LSTM(feature_dim, feature_dim, batch_first=True)
        
        # 注意機構
        self.attention = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x, batch, sizes):
        """
        Args:
            x: 特徴量 [total_atoms, feature_dim]
            batch: バッチインデックス [total_atoms]
            sizes: 各構造の原子数 [batch_size]
        
        Returns:
            プーリングされた特徴量 [batch_size, 2*feature_dim]
        """
        batch_size = sizes.shape[0]
        device = x.device
        
        # 初期状態
        h = torch.zeros(1, batch_size, self.feature_dim, device=device)
        c = torch.zeros(1, batch_size, self.feature_dim, device=device)
        
        # 各ステップで処理
        for _ in range(self.num_steps):
            # LSTMの出力
            q, (h, c) = self.lstm(h.transpose(0, 1), (h, c))
            q = q.squeeze(1)  # [batch_size, feature_dim]
            
            # 注意重みを計算
            attention_scores = torch.zeros_like(batch, dtype=torch.float, device=device)
            
            for i in range(batch_size):
                mask = (batch == i)
                if mask.sum() > 0:
                    x_i = x[mask]  # [num_atoms_i, feature_dim]
                    q_i = q[i:i+1]  # [1, feature_dim]
                    
                    # 注意スコアを計算
                    scores = torch.matmul(x_i, q_i.T).squeeze(-1)  # [num_atoms_i]
                    attention_scores[mask] = torch.softmax(scores, dim=0)
            
            # 重み付き平均を計算
            weighted_x = x * attention_scores.unsqueeze(-1)
            r = scatter(weighted_x, batch, dim=0, reduce='add')  # [batch_size, feature_dim]
            
            # LSTMの入力を更新
            h = r.unsqueeze(0)
        
        # 最終的な表現を計算
        final_repr = torch.cat([q, r], dim=-1)  # [batch_size, 2*feature_dim]
        
        return final_repr


# プーリング関数の辞書
POOLING_FUNCTIONS = {
    'max': max_pool,
    'mean': avr_pool,
    'sum': sum_pool,
    'avr': avr_pool,  # エイリアス
}


def get_pooling_function(pooling_type: str):
    """
    プーリング関数を取得
    
    Args:
        pooling_type: プーリングタイプ
    
    Returns:
        プーリング関数
    """
    if pooling_type in POOLING_FUNCTIONS:
        return POOLING_FUNCTIONS[pooling_type]
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


def create_pooling_layer(pooling_type: str, feature_dim: int = None):
    """
    プーリング層を作成
    
    Args:
        pooling_type: プーリングタイプ
        feature_dim: 特徴量次元（注意機構ベースのプーリングで必要）
    
    Returns:
        プーリング層またはプーリング関数
    """
    if pooling_type in ['max', 'mean', 'sum', 'avr']:
        return get_pooling_function(pooling_type)
    elif pooling_type == 'attention':
        if feature_dim is None:
            raise ValueError("feature_dim is required for attention pooling")
        return AttentionPooling(feature_dim)
    elif pooling_type == 'set2set':
        if feature_dim is None:
            raise ValueError("feature_dim is required for set2set pooling")
        return SetToSetPooling(feature_dim)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
