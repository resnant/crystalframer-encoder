# CrystalFramer Encoder

CrystalFramer Encoderは、事前学習済みのCrystalFramerモデルを結晶構造の埋め込み表現生成に特化したエンコーダとして使用するためのPythonパッケージです。結晶構造を固定次元のベクトル表現に変換し、機械学習タスクや類似性検索に活用できます。

## クイックスタート

```bash
# Dockerコンテナ内で実行
docker run --rm --gpus 1 -it -v $(pwd):/workspace -w /workspace crystalframer:latest bash
cd /workspace/crystalframer-encoder
pip install -e .
```

```python
import crystalframer_encoder as cfe
from pymatgen.core import Structure, Lattice

# エンコーダを読み込み
encoder = cfe.CrystalEncoder.from_pretrained(
    "/workspace/crystalframer-encoder/crystalframer_weight/formation_energy/best.ckpt"
)

# 結晶構造を作成
lattice = Lattice.cubic(4.0)
structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

# エンコード
embedding = encoder.encode_single(structure)
print(f"Embedding shape: {embedding.shape}")  # [1, 128]
```

## 特徴

- **簡単な使用**: 数行のコードで結晶構造をエンコード
- **事前学習済みモデル対応**: CrystalFramerで学習済みのモデルを読み込み可能
- **ハイパーパラメータ自動読み込み**: hparams.yamlファイルからモデル設定を自動取得
- **バッチ処理**: 大量の結晶構造を効率的に処理
- **柔軟な出力**: カスタム埋め込み次元の指定が可能


## インストール

### PyPIからインストール（推奨）

```bash
pip install crystalframer-encoder
```

### GitHubから最新版をインストール

```bash
pip install git+https://github.com/yourusername/crystalframer-encoder.git
```

### 開発用インストール

```bash
git clone https://github.com/yourusername/crystalframer-encoder.git
cd crystalframer-encoder
pip install -e .[dev]
```

### Dockerを使用する場合

```bash
# Dockerコンテナを起動
docker run --rm --gpus 1 -it -v $(pwd):/workspace -w /workspace crystalframer:latest bash

# コンテナ内でパッケージをインストール
pip install crystalframer-encoder
# または開発用
cd /workspace/crystalframer-encoder
pip install -e .
```

**注意**: CrystalFramerの依存関係（PyTorch, PyTorch Geometric等）が正しくインストールされている必要があります。PyTorch Geometricのインストールについては[公式ドキュメント](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)を参照してください。
## 基本的な使用方法

### 1. 事前学習済みモデルの使用

```python
import crystalframer_encoder as cfe

# 方法1: 付属の事前学習済みモデルを使用（Docker環境）
encoder = cfe.CrystalEncoder.from_pretrained(
    "/workspace/crystalframer-encoder/crystalframer_weight/formation_energy/best.ckpt"
)

# 方法2: 独自のモデルファイルを使用
encoder = cfe.CrystalEncoder.from_pretrained("/path/to/your/model.ckpt")

# 方法3: hparams.yamlファイルを明示的に指定
encoder = cfe.CrystalEncoder.from_pretrained(
    "/path/to/model.ckpt", 
    hparams_path="/path/to/hparams.yaml"
)

# 方法4: embedding_dimを手動で指定（モデルの設定を上書き）
encoder = cfe.CrystalEncoder.from_pretrained(
    "/path/to/model.ckpt", 
    embedding_dim=256
)

# 結晶構造を作成
from pymatgen.core import Structure, Lattice
lattice = Lattice.cubic(4.0)
structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

# 単一構造のエンコード
embedding = encoder.encode_single(structure)
print(f"Embedding shape: {embedding.shape}")  # [1, embedding_dim]

# 複数構造のバッチエンコード
structures = [structure1, structure2, structure3, ...]
embeddings = encoder.encode(structures, batch_size=32, show_progress=True)
print(f"Embeddings shape: {embeddings.shape}")  # [num_structures, embedding_dim]
```

### 2. 事前学習済みモデルの入手

事前学習済みモデルは以下の方法で入手できます：

1. **CrystalFramerリポジトリから**: 元のCrystalFramerで学習したモデルを使用
2. **独自に学習**: CrystalFramerを使用して独自のデータセットで学習

サンプルの事前学習済みモデル（JARVIS データセット）:
- Formation Energy予測モデル
- Optical Bandgap予測モデル

**注意**: 大きなモデルファイル（.ckpt）はGitリポジトリには含まれません。別途ダウンロードするか、CrystalFramerで学習する必要があります。

### 3. ハイパーパラメータファイルについて

CrystalFramer Encoderは以下の場所でハイパーパラメータファイルを自動検索します：

- モデルファイルと同じディレクトリの`hparams.yaml`
- モデルファイルの親ディレクトリの`hparams.yaml`
- チェックポイント内の`hyper_parameters`

```python
# ディレクトリ構造例:
# model_directory/
# ├── best.ckpt
# └── hparams.yaml  # 自動的に検出される

encoder = cfe.CrystalEncoder.from_pretrained("/path/to/model_directory/best.ckpt")
```

### 4. スクラッチからの学習

```python
from crystalframer_encoder.utils.params import Params

# パラメータファイルから設定を読み込み
params = Params("/path/to/config.json")

# エンコーダを作成
encoder = cfe.CrystalEncoder(params, embedding_dim=256)

# 通常のPyTorchモデルとして使用
optimizer = torch.optim.Adam(encoder.parameters())
# ... 学習ループ
```

## コマンドラインインターフェース

インストール後、`crystalframer-encode`コマンドが使用可能になります：

```bash
# 単一構造をエンコード
crystalframer-encode structure.cif -m model.ckpt -o embeddings.npz

# 複数構造をエンコード
crystalframer-encode *.cif -m model.ckpt -o embeddings.npz --batch-size 64

# 結果をJSON形式で標準出力に表示
crystalframer-encode structure.cif -m model.ckpt

# ヘルプを表示
crystalframer-encode --help
```

## 応用例

### 結晶構造の類似度分析

```python
# 構造間の類似度を計算
similarity_matrix = encoder.compute_similarity(
    structures1, structures2, metric='cosine'
)

# 最も類似した構造を検索
query_structure = structures[0]
candidates = structures[1:]
similar_structures = encoder.find_similar_structures(
    query_structure, candidates, top_k=5, metric='cosine'
)

for idx, score in similar_structures:
    print(f"Structure {idx}: similarity = {score:.3f}")
```

### 機械学習の特徴量として使用

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 結晶構造を特徴量として抽出
crystal_features = encoder.encode(structures)

# 他の機械学習モデルで使用
model = RandomForestRegressor()
model.fit(crystal_features.numpy(), target_values)

# 予測
predictions = model.predict(crystal_features.numpy())
```

### 結晶構造の可視化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 結晶構造をエンコード
embeddings = encoder.encode(structures)

# t-SNEで次元削減
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings.numpy())

# 可視化
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Crystal Structure Embeddings')
plt.show()
```

### カスタムデータセットでの使用

```python
from torch.utils.data import Dataset, DataLoader

class CrystalDataset(Dataset):
    def __init__(self, structures, labels, encoder):
        self.structures = structures
        self.labels = labels
        self.encoder = encoder
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        label = self.labels[idx]
        
        # エンコーダで特徴量を抽出
        embedding = self.encoder.encode_single(structure)
        
        return embedding.squeeze(0), label
    
    def __len__(self):
        return len(self.structures)

# データセットとデータローダーを作成
dataset = CrystalDataset(structures, labels, encoder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## エンコーダの保存と読み込み

```python
# エンコーダを保存
encoder.save_encoder("my_crystal_encoder.pth")

# 保存されたエンコーダを読み込み
encoder = cfe.CrystalEncoder.load_encoder("my_crystal_encoder.pth")
```

## API リファレンス

### CrystalEncoder

#### 初期化

```python
CrystalEncoder(
    params,                 # Paramsオブジェクト
    embedding_dim=None      # 出力埋め込み次元（Noneの場合はmodel_dimを使用）
)
```

#### クラスメソッド

```python
CrystalEncoder.from_pretrained(
    model_path_or_name,     # モデルファイルのパスまたは事前定義されたモデル名
    hparams_path=None,      # ハイパーパラメータファイルのパス
    embedding_dim=None,     # 出力埋め込み次元
    device='auto',          # 使用デバイス
    **kwargs                # その他のパラメータ
)
```

#### 主要メソッド

- `encode(structures, batch_size=32, show_progress=False)`: 複数構造のバッチエンコード
- `encode_single(structure)`: 単一構造のエンコード
- `compute_similarity(structures1, structures2=None, metric='cosine')`: 構造間類似度計算
- `find_similar_structures(query, candidates, top_k=5, metric='cosine')`: 類似構造検索
- `get_embedding_dim()`: 埋め込み次元を取得
- `save_encoder(path)`: エンコーダの保存
- `load_encoder(path, device='auto')`: エンコーダの読み込み

### Params

```python
from crystalframer_encoder.utils.params import Params

# JSONまたはYAMLファイルから読み込み
params = Params("/path/to/config.json")
params = Params("/path/to/config.yaml")

# 辞書から作成
params = Params.from_dict(config_dict)

# パラメータの取得・設定
value = params.get('key', default_value)
params.set('key', value)
```

## 要件

- Python >= 3.8
- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.0.0
- pymatgen >= 2022.0.0
- PyYAML >= 5.4.0
- その他の依存関係は自動的にインストールされます

## 注意事項

1. **Docker環境でのパス**: Dockerコンテナ内では必ず`/workspace/`で始まるパスを使用してください
2. **メモリ使用量**: 大きな結晶構造や大量のバッチ処理時はメモリ使用量に注意してください
3. **デバイス管理**: GPUを使用する場合、適切なCUDA環境が必要です
4. **依存関係**: PyTorch GeometricやPymatgenなど、必要な依存関係が正しくインストールされていることを確認してください

## トラブルシューティング

### よくある問題

1. **Docker環境でのパスエラー**
   ```
   FileNotFoundError: Model '/mnt/data/work/...' not found.
   ```
   → Docker環境では`/workspace/`で始まるパスを使用してください

2. **インポートエラー**
   ```
   ModuleNotFoundError: No module named 'crystalframer_encoder'
   ```
   → パッケージをインストールしてください: `pip install -e .`

3. **CUDA out of memory**
   → バッチサイズを小さくするか、CPUを使用してください:
   ```python
   encoder = cfe.CrystalEncoder.from_pretrained(model_path, device='cpu')
   ```

4. **ハイパーパラメータファイルが見つからない**
   ```
   Warning: hparams file not found, using default configuration
   ```
   → `hparams_path`パラメータで明示的にファイルパスを指定してください

5. **PyTorch Geometricのインポートエラー**
   → CrystalFramerのDockerイメージを使用するか、PyTorch Geometricを正しくインストールしてください

## テスト

パッケージが正しく動作することを確認するには：

```bash
# Dockerコンテナ内で実行
cd /workspace/crystalframer-encoder
python test_encoder.py  # 基本的なテストスクリプト
python -m pytest tests/  # ユニットテスト（pytestが必要）
```

## ライセンス

このコードはCrystalFramerプロジェクトと同じライセンスに従います。

## 引用

CrystalFramer Encoderを研究で使用する場合は、元のCrystalFramer論文を引用してください：

```bibtex
@inproceedings{ito2025crystalframer,
  title     = {Rethinking the role of frames for SE(3)-invariant crystal structure modeling},
  author    = {Yusei Ito and 
               Tatsunori Taniai and
               Ryo Igarashi and
               Yoshitaka Ushiku and
               Kanta Ono},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR 2025)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=gzxDjnvBDa}
}
