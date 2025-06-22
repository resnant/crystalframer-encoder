"""
CrystalEncoder - 結晶構造エンコーダのメインクラス
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Dict, Any
import os
import copy

from .models.latticeformer import Latticeformer, LatticeformerParams
from .utils.data_utils import structures_to_data_list, batch_structures
from .utils.download import get_model_path
from .utils.params import Params
from .configs.model_configs import get_model_config, DEFAULT_MODEL_CONFIG
from pymatgen.core import Structure
from torch_geometric.loader import DataLoader


class CrystalEncoder(nn.Module):
    """
    CrystalFramerベースの結晶構造エンコーダ
    
    事前学習済みのCrystalFramerモデルから回帰ヘッドを除去し、
    結晶構造の埋め込み表現を生成するエンコーダとして使用可能。
    
    Example:
        # 事前学習済みモデルから初期化
        encoder = CrystalEncoder.from_pretrained("jarvis-formation-energy")
        
        # 結晶構造をエンコード
        embeddings = encoder.encode(structures)
        
        # 単一構造のエンコード
        embedding = encoder.encode_single(structure)
    """
    
    def __init__(self, params, embedding_dim: Optional[int] = None):
        """
        Args:
            params: LatticeformerParams または Params オブジェクト
            embedding_dim: 出力埋め込み次元（Noneの場合はmodel_dimを使用）
        """
        super().__init__()
        
        self.params = params
        self.embedding_dim = embedding_dim or getattr(params, 'model_dim', 128)
        
        # Latticeformerのコア部分を初期化
        self.latticeformer = Latticeformer(params)
        
        # プーリング後の特徴量次元を取得
        pooled_dim = self._get_pooled_dim()
        
        # オプション: 埋め込み次元を調整するための投影層
        if self.embedding_dim != pooled_dim:
            self.projection = nn.Linear(pooled_dim, self.embedding_dim)
        else:
            self.projection = nn.Identity()
            
        self.device_cache = None
        
    def _get_pooled_dim(self) -> int:
        """プーリング後の特徴量次元を取得"""
        if hasattr(self.latticeformer, 'proj_before_pooling'):
            if hasattr(self.latticeformer.proj_before_pooling, '__len__'):
                # Sequential の場合、最初のLinear層の出力次元を取得
                for layer in self.latticeformer.proj_before_pooling:
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
            elif isinstance(self.latticeformer.proj_before_pooling, nn.Linear):
                return self.latticeformer.proj_before_pooling.out_features
        
        # デフォルトはmodel_dim
        return getattr(self.params, 'model_dim', 128)
    
    @classmethod
    def from_pretrained(cls, 
                       model_path_or_name: str,
                       hparams_path: Optional[str] = None,
                       embedding_dim: Optional[int] = None,
                       device: str = 'auto',
                       **kwargs) -> 'CrystalEncoder':
        """
        事前学習済みモデルからエンコーダを初期化
        
        Args:
            model_path_or_name: モデルファイルのパスまたは事前定義されたモデル名
            hparams_path: ハイパーパラメータファイル（hparams.yaml/json）のパス
            embedding_dim: 出力埋め込み次元（Noneの場合はモデルの設定を使用）
            device: 使用デバイス ('auto', 'cpu', 'cuda')
            **kwargs: その他のパラメータ
        
        Returns:
            初期化されたCrystalEncoderインスタンス
        """
        # ファイルパスかモデル名かを判定
        if os.path.exists(model_path_or_name):
            # 直接ファイルパスが指定された場合
            model_path = model_path_or_name
            
            # hparams.yamlファイルを探す
            if hparams_path is None:
                # モデルファイルと同じディレクトリでhparams.yamlを探す
                model_dir = os.path.dirname(model_path)
                potential_hparams_paths = [
                    os.path.join(model_dir, 'hparams.yaml'),
                    os.path.join(model_dir, 'hparams.yml'),
                    os.path.join(model_dir, 'hparams.json'),
                    os.path.join(model_dir, '..', 'hparams.yaml'),  # 一つ上のディレクトリ
                    os.path.join(model_dir, '..', 'hparams.yml'),
                    os.path.join(model_dir, '..', 'hparams.json'),
                ]
                
                for path in potential_hparams_paths:
                    if os.path.exists(path):
                        hparams_path = path
                        break
            
            # hparams.yamlから設定を読み込み
            if hparams_path and os.path.exists(hparams_path):
                print(f"Loading hyperparameters from: {hparams_path}")
                try:
                    params = Params(hparams_path)
                    print(f"Loaded hyperparameters: {list(params.keys())}")
                except Exception as e:
                    print(f"Warning: Failed to load hparams file: {e}")
                    params = Params.from_dict(DEFAULT_MODEL_CONFIG.copy())
            else:
                print("Warning: hparams file not found, using default configuration")
                params = Params.from_dict(DEFAULT_MODEL_CONFIG.copy())
            
            # kwargsで設定を更新
            params.update_dict(kwargs)
        else:
            # 事前定義されたモデル名の場合
            try:
                config = get_model_config(model_path_or_name)
                config.update(kwargs)
                params = Params.from_dict(config)
                # モデルファイルのパスを取得（ダウンロードは無効化）
                model_path = get_model_path(model_path_or_name, auto_download=False)
            except (ValueError, FileNotFoundError) as e:
                raise FileNotFoundError(
                    f"Model '{model_path_or_name}' not found. "
                    f"Please provide a valid file path or ensure the model file exists. "
                    f"Original error: {e}"
                )
        
        # チェックポイントを読み込んでハイパーパラメータを確認
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # チェックポイント内にhyper_parametersがある場合はそれを使用
            if 'hyper_parameters' in checkpoint:
                checkpoint_hparams = checkpoint['hyper_parameters']
                print(f"Found hyperparameters in checkpoint: {list(checkpoint_hparams.keys())}")
                # チェックポイントのハイパーパラメータで設定を更新
                params.update_dict(checkpoint_hparams)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")
        
        # embedding_dimの処理
        if embedding_dim is None:
            # モデルの設定から埋め込み次元を取得
            model_embedding_dim = params.get('embedding_dim')
            if model_embedding_dim is not None:
                if isinstance(model_embedding_dim, list):
                    embedding_dim = model_embedding_dim[0] if model_embedding_dim else params.get('model_dim', 128)
                else:
                    embedding_dim = model_embedding_dim
            else:
                embedding_dim = params.get('model_dim', 128)
        
        # エンコーダを初期化
        encoder = cls(params, embedding_dim)
        
        # 学習済み重みを読み込み
        state_dict = checkpoint['state_dict']
        
        # RegressionModelの重みからLatticeformerの重みを抽出
        latticeformer_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                # 'model.' プレフィックスを除去
                new_key = key[6:]  # 'model.' を除去
                latticeformer_state_dict[new_key] = value
        
        # 重みを読み込み（strict=Falseで回帰ヘッド部分は無視）
        missing_keys, unexpected_keys = encoder.latticeformer.load_state_dict(
            latticeformer_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
        
        # デバイスを設定
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = encoder.to(device)
        encoder.device_cache = device
        
        print(f"CrystalEncoder loaded from: {model_path}")
        print(f"Device: {device}")
        print(f"Embedding dimension: {encoder.embedding_dim}")
        
        return encoder
    
    def forward(self, data, return_pooled_only: bool = True):
        """
        結晶構造データをエンコードして埋め込み表現を生成
        
        Args:
            data: torch_geometric.data.Data または Batch オブジェクト
            return_pooled_only: プーリング後の特徴量のみを返すかどうか
        
        Returns:
            torch.Tensor: 結晶構造の埋め込み表現 [batch_size, embedding_dim]
        """
        # デバイスを自動調整
        if self.device_cache is not None and data.x.device != self.device_cache:
            data = data.to(self.device_cache)
        
        # Latticeformerのforward処理を部分的に実行
        x = data.x
        pos = data.pos
        batch = data.batch
        trans = data.trans_vec
        sizes = data.sizes
        
        # 入力埋め込み
        x = self.latticeformer.input_embeddings(x)
        
        # エンコーダ処理
        x = self.latticeformer.encoder(x, pos, batch, trans, sizes)
        
        if not return_pooled_only:
            # プーリング前の特徴量も返す場合
            pre_pooled = x.clone()
        
        # プーリング前の投影
        x = self.latticeformer.proj_before_pooling(x)
        
        # プーリング
        if self.latticeformer.pooling.startswith("pma"):
            pooled = self.latticeformer.pooling_layer(x, batch, sizes.shape[0])
        else:
            pooled = self.latticeformer.pooling_layer(x, batch, sizes)
        
        # 最終的な埋め込み次元に投影
        embeddings = self.projection(pooled)
        
        if return_pooled_only:
            return embeddings
        else:
            return {
                'embeddings': embeddings,
                'pre_pooled': pre_pooled,
                'pooled': pooled
            }
    
    def encode(self, structures: List[Structure], 
               batch_size: int = 32,
               show_progress: bool = False) -> torch.Tensor:
        """
        複数の結晶構造をバッチ処理でエンコード
        
        Args:
            structures: pymatgen.Structure オブジェクトのリスト
            batch_size: バッチサイズ
            show_progress: プログレスバーを表示するかどうか
        
        Returns:
            torch.Tensor: 埋め込み表現 [num_structures, embedding_dim]
        """
        if not structures:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        
        # 結晶構造をData形式に変換
        data_list = structures_to_data_list(structures)
        
        # DataLoaderでバッチ処理
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        
        embeddings_list = []
        self.eval()
        
        if show_progress:
            from tqdm import tqdm
            loader = tqdm(loader, desc="Encoding structures")
        
        with torch.no_grad():
            for batch_data in loader:
                if self.device_cache is not None:
                    batch_data = batch_data.to(self.device_cache)
                embeddings = self.forward(batch_data)
                embeddings_list.append(embeddings.cpu())
        
        return torch.cat(embeddings_list, dim=0)
    
    def encode_single(self, structure: Structure) -> torch.Tensor:
        """
        単一の結晶構造をエンコード
        
        Args:
            structure: pymatgen.Structure オブジェクト
        
        Returns:
            torch.Tensor: 埋め込み表現 [1, embedding_dim]
        """
        # バッチ形式に変換
        batch_data = batch_structures([structure])
        
        self.eval()
        with torch.no_grad():
            if self.device_cache is not None:
                batch_data = batch_data.to(self.device_cache)
            embedding = self.forward(batch_data)
        
        return embedding.cpu()
    
    def get_embedding_dim(self) -> int:
        """埋め込み次元を取得"""
        return self.embedding_dim
    
    def save_encoder(self, save_path: str):
        """エンコーダの重みを保存"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.params.__dict__ if hasattr(self.params, '__dict__') else vars(self.params),
            'embedding_dim': self.embedding_dim
        }, save_path)
        print(f"CrystalEncoder saved to {save_path}")
    
    @classmethod
    def load_encoder(cls, save_path: str, device: str = 'auto') -> 'CrystalEncoder':
        """保存されたエンコーダを読み込み"""
        checkpoint = torch.load(save_path, map_location='cpu')
        
        config = checkpoint['config']
        embedding_dim = checkpoint.get('embedding_dim', None)
        
        # Paramsオブジェクトを作成
        params = type('Params', (), config)()
        
        encoder = cls(params, embedding_dim)
        encoder.load_state_dict(checkpoint['state_dict'])
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = encoder.to(device)
        encoder.device_cache = device
        
        return encoder
    
    def compute_similarity(self, 
                          structures1: List[Structure],
                          structures2: Optional[List[Structure]] = None,
                          metric: str = 'cosine') -> torch.Tensor:
        """
        結晶構造間の類似度を計算
        
        Args:
            structures1: 第1の結晶構造リスト
            structures2: 第2の結晶構造リスト（Noneの場合はstructures1同士）
            metric: 類似度メトリック ('cosine', 'euclidean')
        
        Returns:
            類似度行列 [len(structures1), len(structures2)]
        """
        embeddings1 = self.encode(structures1)
        
        if structures2 is None:
            embeddings2 = embeddings1
        else:
            embeddings2 = self.encode(structures2)
        
        if metric == 'cosine':
            # コサイン類似度
            embeddings1_norm = F.normalize(embeddings1, dim=1)
            embeddings2_norm = F.normalize(embeddings2, dim=1)
            similarity = torch.matmul(embeddings1_norm, embeddings2_norm.T)
        elif metric == 'euclidean':
            # ユークリッド距離（負の値で類似度として扱う）
            dist = torch.cdist(embeddings1, embeddings2, p=2)
            similarity = -dist
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def find_similar_structures(self,
                               query_structure: Structure,
                               candidate_structures: List[Structure],
                               top_k: int = 5,
                               metric: str = 'cosine') -> List[tuple]:
        """
        クエリ構造に類似した構造を検索
        
        Args:
            query_structure: クエリ構造
            candidate_structures: 候補構造のリスト
            top_k: 上位k個を返す
            metric: 類似度メトリック
        
        Returns:
            (index, similarity_score) のタプルのリスト
        """
        similarity = self.compute_similarity([query_structure], candidate_structures, metric)
        similarity_scores = similarity[0]  # 最初の行（クエリ構造との類似度）
        
        # 上位k個のインデックスを取得
        top_indices = torch.topk(similarity_scores, min(top_k, len(candidate_structures))).indices
        
        results = []
        for idx in top_indices:
            results.append((idx.item(), similarity_scores[idx].item()))
        
        return results
