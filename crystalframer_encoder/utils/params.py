"""
パラメータ管理ユーティリティ
"""

import os
import json
import yaml
from typing import Dict, Any, Optional


class Params:
    """
    JSONまたはYAMLファイルからハイパーパラメータを読み込むクラス
    
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # パラメータ値を変更
    ```
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Args:
            file_path: JSONまたはYAMLファイルのパス（Noneの場合は空のParamsオブジェクト）
        """
        if file_path is not None:
            self.load(file_path)
    
    def load(self, file_path: str):
        """ファイルからパラメータを読み込み"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parameter file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .json, .yaml, or .yml")
        
        self.__dict__.update(params)
    
    def save(self, file_path: str):
        """パラメータをファイルに保存"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # ディレクトリを作成
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_ext == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, indent=4, ensure_ascii=False)
        elif file_ext in ['.yaml', '.yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .json, .yaml, or .yml")
    
    def update(self, file_path: str):
        """ファイルからパラメータを更新"""
        self.load(file_path)
    
    def update_dict(self, params_dict: Dict[str, Any]):
        """辞書からパラメータを更新"""
        self.__dict__.update(params_dict)
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'Params':
        """辞書からParamsオブジェクトを作成"""
        params = cls()
        params.__dict__.update(params_dict)
        return params
    
    @property
    def dict(self) -> Dict[str, Any]:
        """辞書形式でのアクセスを提供"""
        return self.__dict__
    
    def get(self, key: str, default: Any = None) -> Any:
        """パラメータを取得（デフォルト値付き）"""
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any):
        """パラメータを設定"""
        setattr(self, key, value)
    
    def keys(self):
        """パラメータのキー一覧を取得"""
        return self.__dict__.keys()
    
    def items(self):
        """パラメータのキー・値ペアを取得"""
        return self.__dict__.items()
    
    def __repr__(self) -> str:
        return f"Params({self.__dict__})"
    
    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)
