"""
データ変換ユーティリティ
pymatgen.Structure から torch_geometric.Data への変換など
"""

import torch
import numpy as np
from typing import List, Union, Optional
from pymatgen.core import Structure
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


def generate_site_species_vector(structure: Structure, atom_num_upper: int = 98) -> torch.Tensor:
    """
    結晶構造から原子種のワンホットベクトルを生成
    
    Args:
        structure: pymatgen.Structure オブジェクト
        atom_num_upper: 原子番号の上限
    
    Returns:
        原子種のワンホットベクトル [num_atoms, atom_num_upper]
    """
    num_atoms = len(structure)
    atom_features = torch.zeros(num_atoms, atom_num_upper, dtype=torch.float32)
    
    for i, site in enumerate(structure):
        # 各サイトの元素を取得
        for element, occupancy in site.species.items():
            atomic_num = element.Z
            if atomic_num <= atom_num_upper:
                atom_features[i, atomic_num - 1] = occupancy
    
    return atom_features


def structure_to_data(structure: Structure, 
                     atom_num_upper: int = 98,
                     material_id: Optional[Union[str, int]] = None) -> Data:
    """
    pymatgen.Structure を torch_geometric.Data に変換
    
    Args:
        structure: pymatgen.Structure オブジェクト
        atom_num_upper: 原子番号の上限
        material_id: 材料ID（オプション）
    
    Returns:
        torch_geometric.Data オブジェクト
    """
    # 原子座標
    atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
    
    # 原子特徴量（ワンホットベクトル）
    atom_features = generate_site_species_vector(structure, atom_num_upper)
    
    # torch_geometric.Data オブジェクトを作成
    data = Data(
        x=atom_features,
        pos=atom_pos,
        y=None  # ラベルは後で設定
    )
    
    # 格子ベクトル（周期境界条件用）
    data.trans_vec = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    
    # 原子数
    data.sizes = torch.tensor([atom_pos.shape[0]], dtype=torch.long)
    
    # 材料ID
    if material_id is not None:
        data.material_id = material_id
    
    return data


def structures_to_data_list(structures: List[Structure],
                           atom_num_upper: int = 98,
                           material_ids: Optional[List[Union[str, int]]] = None) -> List[Data]:
    """
    複数のpymatgen.Structureをtorch_geometric.Dataのリストに変換
    
    Args:
        structures: pymatgen.Structure オブジェクトのリスト
        atom_num_upper: 原子番号の上限
        material_ids: 材料IDのリスト（オプション）
    
    Returns:
        torch_geometric.Data オブジェクトのリスト
    """
    data_list = []
    
    for i, structure in enumerate(structures):
        material_id = material_ids[i] if material_ids else i
        data = structure_to_data(structure, atom_num_upper, material_id)
        data_list.append(data)
    
    return data_list


def create_dataloader(structures: List[Structure],
                     batch_size: int = 32,
                     shuffle: bool = False,
                     atom_num_upper: int = 98,
                     material_ids: Optional[List[Union[str, int]]] = None,
                     **dataloader_kwargs) -> DataLoader:
    """
    結晶構造のリストからDataLoaderを作成
    
    Args:
        structures: pymatgen.Structure オブジェクトのリスト
        batch_size: バッチサイズ
        shuffle: シャッフルするかどうか
        atom_num_upper: 原子番号の上限
        material_ids: 材料IDのリスト（オプション）
        **dataloader_kwargs: DataLoaderの追加引数
    
    Returns:
        torch_geometric.loader.DataLoader
    """
    data_list = structures_to_data_list(structures, atom_num_upper, material_ids)
    
    return DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs
    )


def batch_structures(structures: List[Structure],
                    atom_num_upper: int = 98,
                    material_ids: Optional[List[Union[str, int]]] = None) -> Batch:
    """
    複数の結晶構造を1つのバッチにまとめる
    
    Args:
        structures: pymatgen.Structure オブジェクトのリスト
        atom_num_upper: 原子番号の上限
        material_ids: 材料IDのリスト（オプション）
    
    Returns:
        torch_geometric.data.Batch オブジェクト
    """
    data_list = structures_to_data_list(structures, atom_num_upper, material_ids)
    return Batch.from_data_list(data_list)


def validate_structure(structure: Structure) -> bool:
    """
    結晶構造の妥当性をチェック
    
    Args:
        structure: pymatgen.Structure オブジェクト
    
    Returns:
        妥当性（True/False）
    """
    try:
        # 基本的なチェック
        if len(structure) == 0:
            return False
        
        # 原子座標が有効かチェック
        coords = structure.cart_coords
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            return False
        
        # 格子ベクトルが有効かチェック
        lattice_matrix = structure.lattice.matrix
        if np.any(np.isnan(lattice_matrix)) or np.any(np.isinf(lattice_matrix)):
            return False
        
        # 格子の体積が正の値かチェック
        if structure.lattice.volume <= 0:
            return False
        
        return True
        
    except Exception:
        return False


def filter_valid_structures(structures: List[Structure]) -> List[Structure]:
    """
    有効な結晶構造のみをフィルタリング
    
    Args:
        structures: pymatgen.Structure オブジェクトのリスト
    
    Returns:
        有効な結晶構造のリスト
    """
    valid_structures = []
    
    for structure in structures:
        if validate_structure(structure):
            valid_structures.append(structure)
    
    return valid_structures


def get_structure_info(structure: Structure) -> dict:
    """
    結晶構造の基本情報を取得
    
    Args:
        structure: pymatgen.Structure オブジェクト
    
    Returns:
        構造情報の辞書
    """
    return {
        "formula": structure.composition.reduced_formula,
        "num_atoms": len(structure),
        "volume": structure.lattice.volume,
        "density": structure.density,
        "space_group": structure.get_space_group_info()[1],
        "lattice_parameters": {
            "a": structure.lattice.a,
            "b": structure.lattice.b,
            "c": structure.lattice.c,
            "alpha": structure.lattice.alpha,
            "beta": structure.lattice.beta,
            "gamma": structure.lattice.gamma,
        }
    }


def print_structure_summary(structures: List[Structure]) -> None:
    """
    結晶構造リストの要約を表示
    
    Args:
        structures: pymatgen.Structure オブジェクトのリスト
    """
    if not structures:
        print("No structures provided.")
        return
    
    print(f"Total structures: {len(structures)}")
    
    # 原子数の統計
    num_atoms_list = [len(s) for s in structures]
    print(f"Number of atoms - Min: {min(num_atoms_list)}, Max: {max(num_atoms_list)}, Mean: {np.mean(num_atoms_list):.1f}")
    
    # 組成の統計
    formulas = [s.composition.reduced_formula for s in structures]
    unique_formulas = set(formulas)
    print(f"Unique compositions: {len(unique_formulas)}")
    
    # 最も一般的な組成を表示
    from collections import Counter
    formula_counts = Counter(formulas)
    most_common = formula_counts.most_common(5)
    print("Most common compositions:")
    for formula, count in most_common:
        print(f"  {formula}: {count}")
