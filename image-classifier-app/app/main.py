# 1. streamlitで画像アップロードUIを作成
# 2. 画像を前処理（utils内の関数を使用）
# 3. 学習済みモデルをtorch.loadで読み込み
# 4. 推論結果（クラス）を表示
# 5. クラスラベルと画像を一緒に表示


# app/main.py

import streamlit as st  # StreamlitでUI構築
from PIL import Image  # アップロード画像を扱うため
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import os

# 相対インポート用（utilsのpreprocessing関数を使うため）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import preprocess_image  # 前処理関数をインポート

# ==========================
# クラスラベルの定義（CIFAR-10）
# ==========================
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ==========================
# CNNモデル定義（学習時と同じ構造にする必要あり）
# ==========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================
# Streamlit UIレイアウト
# ==========================
st.set_page_config(page_title="画像分類AI", layout="centered")
st.title("🧠 画像分類AIアプリ")
st.markdown("アップロードした画像を分類します。対応クラス: `CIFAR-10`")

# ==========================
# モデル読み込み（学習済モデル）
# ==========================
model = SimpleCNN()
model.load_state_dict(torch.load("models/cifar10_cnn.pt", map_location=torch.device("cpu")))
model.eval()  # 推論モードに切り替え

# ==========================
# ファイルアップロードUI
# ==========================
uploaded_file = st.file_uploader("画像ファイルを選択（例：.jpg, .png）", type=["jpg", "png"])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード画像", use_column_width=True)

    # ボタンで予測実行
    if st.button("分類する"):
        with st.spinner("画像を分類中..."):
            # 前処理（utilsの関数を使用）
            input_tensor = preprocess_image(image)

            # 推論実行
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = classes[predicted.item()]

            # 結果表示
            st.success(f"✅ 予測クラス: **{predicted_class}**")
