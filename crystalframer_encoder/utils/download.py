"""
事前学習済みモデルのダウンロード機能
"""

import os
import hashlib
import requests
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import shutil

from ..configs.model_configs import (
    get_model_info, 
    get_cache_dir, 
    update_model_config,
    list_available_models
)


def calculate_sha256(file_path: str) -> str:
    """
    ファイルのSHA256ハッシュを計算
    
    Args:
        file_path: ファイルパス
    
    Returns:
        SHA256ハッシュ値
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # ファイルを小さなチャンクに分けて読み込み
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file(url: str, 
                 output_path: str, 
                 expected_size: Optional[int] = None,
                 chunk_size: int = 8192) -> bool:
    """
    URLからファイルをダウンロード
    
    Args:
        url: ダウンロードURL
        output_path: 出力ファイルパス
        expected_size: 期待されるファイルサイズ（バイト）
        chunk_size: チャンクサイズ
    
    Returns:
        ダウンロード成功フラグ
    """
    try:
        # ディレクトリを作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # HTTPリクエスト
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # ファイルサイズを取得
        total_size = int(response.headers.get('content-length', 0))
        if expected_size and total_size != expected_size:
            print(f"Warning: Expected size {expected_size}, but got {total_size}")
        
        # プログレスバー付きでダウンロード
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        # 失敗した場合は部分的なファイルを削除
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def verify_file(file_path: str, expected_sha256: str) -> bool:
    """
    ファイルの整合性を検証
    
    Args:
        file_path: ファイルパス
        expected_sha256: 期待されるSHA256ハッシュ
    
    Returns:
        検証結果
    """
    if not os.path.exists(file_path):
        return False
    
    actual_sha256 = calculate_sha256(file_path)
    return actual_sha256 == expected_sha256


def download_pretrained_model(model_name: str, 
                             cache_dir: Optional[str] = None,
                             force_download: bool = False) -> str:
    """
    事前学習済みモデルをダウンロード
    
    Args:
        model_name: モデル名
        cache_dir: キャッシュディレクトリ（Noneの場合はデフォルト）
        force_download: 強制再ダウンロード
    
    Returns:
        ダウンロードされたファイルのパス
    """
    # モデル情報を取得
    model_info = get_model_info(model_name)
    
    # キャッシュディレクトリを設定
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    # ファイルパスを構築
    filename = f"{model_name}.ckpt"
    file_path = os.path.join(cache_dir, filename)
    
    # 既存ファイルの確認
    if os.path.exists(file_path) and not force_download:
        # ハッシュ値を確認
        if verify_file(file_path, model_info["sha256"]):
            print(f"Model '{model_name}' already exists and verified: {file_path}")
            update_model_config(model_name, file_path)
            return file_path
        else:
            print(f"Existing file corrupted, re-downloading...")
    
    # ダウンロード
    url = model_info["url"]
    expected_size = model_info.get("file_size")
    
    print(f"Downloading model '{model_name}' from {url}")
    
    # URLがローカルファイルパスの場合
    if url.startswith("file://"):
        local_path = url[7:]  # "file://" を除去
        if os.path.exists(local_path):
            print(f"Copying from local path: {local_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy2(local_path, file_path)
        else:
            raise FileNotFoundError(f"Local file not found: {local_path}")
    else:
        # HTTPダウンロード
        success = download_file(url, file_path, expected_size)
        if not success:
            raise RuntimeError(f"Failed to download model '{model_name}'")
    
    # ファイルの整合性を検証
    if not verify_file(file_path, model_info["sha256"]):
        os.remove(file_path)
        raise RuntimeError(f"Downloaded file corrupted for model '{model_name}'")
    
    print(f"Model '{model_name}' downloaded successfully: {file_path}")
    
    # 設定を更新
    update_model_config(model_name, file_path)
    
    return file_path


def get_model_path(model_name: str, 
                  cache_dir: Optional[str] = None,
                  auto_download: bool = True) -> str:
    """
    モデルファイルのパスを取得（必要に応じてダウンロード）
    
    Args:
        model_name: モデル名
        cache_dir: キャッシュディレクトリ
        auto_download: 自動ダウンロードするかどうか
    
    Returns:
        モデルファイルのパス
    """
    model_info = get_model_info(model_name)
    
    # ローカルパスが設定されている場合
    if model_info.get("local_path") and os.path.exists(model_info["local_path"]):
        return model_info["local_path"]
    
    # キャッシュディレクトリを確認
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    filename = f"{model_name}.ckpt"
    file_path = os.path.join(cache_dir, filename)
    
    # ファイルが存在し、検証が通る場合
    if os.path.exists(file_path) and verify_file(file_path, model_info["sha256"]):
        update_model_config(model_name, file_path)
        return file_path
    
    # 自動ダウンロード
    if auto_download:
        return download_pretrained_model(model_name, cache_dir)
    else:
        raise FileNotFoundError(
            f"Model '{model_name}' not found. "
            f"Set auto_download=True or manually download the model."
        )


def clear_cache(cache_dir: Optional[str] = None) -> None:
    """
    キャッシュディレクトリをクリア
    
    Args:
        cache_dir: キャッシュディレクトリ（Noneの場合はデフォルト）
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache cleared: {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")


def list_cached_models(cache_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    キャッシュされたモデルのリストを取得
    
    Args:
        cache_dir: キャッシュディレクトリ
    
    Returns:
        キャッシュされたモデルの情報
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    cached_models = {}
    
    if not os.path.exists(cache_dir):
        return cached_models
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('.ckpt'):
            model_name = filename[:-5]  # .ckpt を除去
            file_path = os.path.join(cache_dir, filename)
            
            # ファイル情報を取得
            stat = os.stat(file_path)
            cached_models[model_name] = {
                "path": file_path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
    
    return cached_models


def main():
    """
    コマンドライン用のメイン関数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="CrystalFramer model downloader")
    parser.add_argument("command", choices=["download", "list", "clear", "verify"])
    parser.add_argument("--model", type=str, help="Model name to download")
    parser.add_argument("--cache-dir", type=str, help="Cache directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("Available models:")
        for name, desc in list_available_models().items():
            print(f"  {name}: {desc}")
        
        print("\nCached models:")
        cached = list_cached_models(args.cache_dir)
        for name, info in cached.items():
            size_mb = info["size"] / (1024 * 1024)
            print(f"  {name}: {size_mb:.1f} MB")
    
    elif args.command == "download":
        if not args.model:
            print("Error: --model is required for download command")
            return
        
        try:
            path = download_pretrained_model(
                args.model, 
                args.cache_dir, 
                args.force
            )
            print(f"Downloaded: {path}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.command == "clear":
        clear_cache(args.cache_dir)
    
    elif args.command == "verify":
        if not args.model:
            print("Error: --model is required for verify command")
            return
        
        try:
            model_info = get_model_info(args.model)
            path = get_model_path(args.model, args.cache_dir, auto_download=False)
            
            if verify_file(path, model_info["sha256"]):
                print(f"✓ Model '{args.model}' verified successfully")
            else:
                print(f"✗ Model '{args.model}' verification failed")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
