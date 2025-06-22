"""
CrystalFramer Encoder - 結晶構造エンコーダパッケージ

事前学習済みのCrystalFramerモデルを使用して結晶構造の埋め込み表現を生成するためのパッケージ
"""

from .encoder import CrystalEncoder
from .configs.model_configs import list_available_models
from .utils.download import download_pretrained_model, list_cached_models, clear_cache

__version__ = "0.1.0"
__author__ = "CrystalFramer Team"
__email__ = "your-email@example.com"

# メインのエクスポート
__all__ = [
    "CrystalEncoder",
    "list_available_models", 
    "download_pretrained_model",
    "list_cached_models",
    "clear_cache",
]

# パッケージレベルの便利関数
def create_encoder(model_name=None, **kwargs):
    """
    エンコーダを作成する便利関数
    
    Args:
        model_name: 事前学習済みモデル名（Noneの場合はスクラッチ）
        **kwargs: その他のパラメータ
    
    Returns:
        CrystalEncoder インスタンス
    """
    if model_name is None:
        return CrystalEncoder(**kwargs)
    else:
        return CrystalEncoder.from_pretrained(model_name, **kwargs)


def show_available_models():
    """利用可能なモデルを表示"""
    models = list_available_models()
    print("Available pretrained models:")
    print("-" * 50)
    for name, description in models.items():
        print(f"{name:25} : {description}")


def show_cached_models():
    """キャッシュされたモデルを表示"""
    cached = list_cached_models()
    if not cached:
        print("No cached models found.")
        return
    
    print("Cached models:")
    print("-" * 50)
    for name, info in cached.items():
        size_mb = info["size"] / (1024 * 1024)
        print(f"{name:25} : {size_mb:.1f} MB")


# パッケージ情報
def get_package_info():
    """パッケージ情報を取得"""
    return {
        "name": "crystalframer-encoder",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Crystal structure encoder based on CrystalFramer",
    }


# バージョン情報を表示（デバッグ時のみ）
import os
if os.environ.get("CRYSTALFRAMER_DEBUG"):
    print(f"CrystalFramer Encoder v{__version__}")
    print("For help, use: help(crystalframer_encoder)")
