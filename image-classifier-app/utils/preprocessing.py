# 1. PIL画像をTensorに変換
# 2. 正規化
# 3. バッチ次元を追加してモデルへ送れる形式にする


# utils/preprocessing.py

from PIL import Image  # アップロード画像の読み込みに使用
import torchvision.transforms as transforms
import torch

# ==========================
# 前処理関数：画像をTensor形式に変換＋標準化
# ==========================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    入力画像をモデルが受け取れるように前処理する。

    パラメータ:
    - image (PIL.Image): アップロードされた画像

    戻り値:
    - torch.Tensor: モデル推論用の前処理済みTensor
    """

    # 画像変換パイプラインを定義
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10のサイズにリサイズ
        transforms.ToTensor(),        # PIL画像をTensorに変換（[0, 1]）
        transforms.Normalize(         # データを正規化（平均0.5, 分散0.5）
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # 変換を適用してTensorに
    image_tensor = transform(image)

    # モデルはバッチ入力を期待しているので、次元を追加（[1, C, H, W]）
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
