"""
事前学習済みモデルの設定とメタデータ
"""

from typing import Dict, Any, Optional
import os

# 事前学習済みモデルの設定
PRETRAINED_MODELS: Dict[str, Dict[str, Any]] = {
    "jarvis-formation-energy": {
        "description": "CrystalFramer trained on JARVIS formation energy dataset",
        "url": "https://your-private-repo.com/models/jarvis_formation_energy.ckpt",
        "local_path": None,  # ローカルパスが指定されている場合
        "config": {
            "model_dim": 128,
            "num_layers": 4,
            "head_num": 8,
            "ff_dim": 512,
            "embedding_dim": [128],
            "pooling": "avr",
            "frame_method": "max",
            "value_pe_angle_real": 64,
            "value_pe_dist_real": 64,
            "scale_real": [1.4],
            "domain": "real",
            "encoder_name": "latticeformer",
        },
        "sha256": "abc123def456...",  # チェックサム（実際の値に置き換え）
        "file_size": 50000000,  # バイト単位
    },
    "mp-bandgap": {
        "description": "CrystalFramer trained on Materials Project bandgap dataset",
        "url": "https://your-private-repo.com/models/mp_bandgap.ckpt",
        "local_path": None,
        "config": {
            "model_dim": 128,
            "num_layers": 4,
            "head_num": 8,
            "ff_dim": 512,
            "embedding_dim": [128],
            "pooling": "avr",
            "frame_method": "max",
            "value_pe_angle_real": 64,
            "value_pe_dist_real": 64,
            "scale_real": [1.4],
            "domain": "real",
            "encoder_name": "latticeformer",
        },
        "sha256": "def456ghi789...",
        "file_size": 50000000,
    },
    "oqmd-stability": {
        "description": "CrystalFramer trained on OQMD stability dataset",
        "url": "https://your-private-repo.com/models/oqmd_stability.ckpt",
        "local_path": None,
        "config": {
            "model_dim": 128,
            "num_layers": 4,
            "head_num": 8,
            "ff_dim": 512,
            "embedding_dim": [128],
            "pooling": "avr",
            "frame_method": "max",
            "value_pe_angle_real": 64,
            "value_pe_dist_real": 64,
            "scale_real": [1.4],
            "domain": "real",
            "encoder_name": "latticeformer",
        },
        "sha256": "ghi789jkl012...",
        "file_size": 50000000,
    },
}

# デフォルトのモデル設定（スクラッチ学習用）
DEFAULT_MODEL_CONFIG = {
    "model_dim": 128,
    "num_layers": 4,
    "head_num": 8,
    "ff_dim": 512,
    "embedding_dim": [128],
    "pooling": "avr",
    "pre_pooling_op": "no",
    "norm_type": "no",
    "dropout": 0.0,
    "t_fixup_init": True,
    "t_activation": "relu",
    "domain": "real",
    "frame_method": "max",
    "value_pe_angle_real": 64,
    "value_pe_angle_wscale": 4.0,
    "value_pe_angle_coef": 1.0,
    "value_pe_dist_real": 64,
    "value_pe_dist_coef": 1.0,
    "value_pe_dist_wscale": 1.0,
    "value_pe_dist_max": -10.0,
    "scale_real": [1.4],
    "scale_reci": [2.2],
    "gauss_lb_real": 0.5,
    "gauss_lb_reci": 0.5,
    "normalize_gauss": True,
    "positive_func_beta": 0.1,
    "gauss_state": "q",
    "cos_abs": 0,
    "lattice_range": 2,
    "minimum_range": True,
    "adaptive_cutoff_sigma": -3.5,
    "symm_break_noise": 1e-5,
    "use_cuda_code": True,
    "encoder_name": "latticeformer",
    "targets": "dummy",  # エンコーダでは使用しないが、互換性のため
}


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    モデル設定を取得
    
    Args:
        model_name: 事前学習済みモデル名（Noneの場合はデフォルト設定）
    
    Returns:
        モデル設定辞書
    """
    if model_name is None:
        return DEFAULT_MODEL_CONFIG.copy()
    
    if model_name not in PRETRAINED_MODELS:
        available_models = list(PRETRAINED_MODELS.keys())
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available models: {available_models}"
        )
    
    return PRETRAINED_MODELS[model_name]["config"].copy()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    モデル情報を取得
    
    Args:
        model_name: 事前学習済みモデル名
    
    Returns:
        モデル情報辞書
    """
    if model_name not in PRETRAINED_MODELS:
        available_models = list(PRETRAINED_MODELS.keys())
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available models: {available_models}"
        )
    
    return PRETRAINED_MODELS[model_name].copy()


def list_available_models() -> Dict[str, str]:
    """
    利用可能なモデルのリストを取得
    
    Returns:
        {model_name: description} の辞書
    """
    return {
        name: info["description"] 
        for name, info in PRETRAINED_MODELS.items()
    }


def update_model_config(model_name: str, local_path: str) -> None:
    """
    モデル設定のローカルパスを更新
    
    Args:
        model_name: モデル名
        local_path: ローカルファイルパス
    """
    if model_name in PRETRAINED_MODELS:
        PRETRAINED_MODELS[model_name]["local_path"] = local_path


def get_cache_dir() -> str:
    """
    キャッシュディレクトリを取得
    
    Returns:
        キャッシュディレクトリパス
    """
    # 環境変数で指定されている場合はそれを使用
    cache_dir = os.environ.get("CRYSTALFRAMER_CACHE_DIR")
    if cache_dir:
        return cache_dir
    
    # デフォルトはホームディレクトリ下
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, ".cache", "crystalframer_encoder")
