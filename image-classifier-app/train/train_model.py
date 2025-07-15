# 3. CNNモデル構築（Conv -> ReLU -> Pool -> FC）
# 4. 損失関数とOptimizer設定
# 5. エポックでループして学習
# 6. モデルを保存（torch.save）

# train/train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# TIPS: インポートしたライブラリたちの紹介
# 📌PyTorch
# Pythonの機械学習用フレームワーク
# この中に機械学習に有用なライブラリたちが用意されている
# |
# |----📌torch
# |     機械学習のためのオープンソース計算ライブラリ
# |     NumPyの構造を模している
# |     Tensor処理の数学関数が実装されている
# |     |
# |     |----📌torch.nn
# |     |     ニューラルネットワークを構築するためのモデルの定義に使用するライブラリ
# |     |
# |     |----📌torch.potim
# |           学習のパラメータを最適化するアルゴリズムが実装されているライブラリ
# |
# |----📌torchvision
# |     データを様々な形式に変換する関数が実装されたライブラリ
# |     |
# |     |----📌torchvision.transforms 

# =======================
# 1. デバイス（GPU or CPU）設定
# =======================

# GPU（使用できないならCPU）を使用するように設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# 2. 画像の前処理とデータローダー設定（CIFAR-10）
# =======================
# TIPS: CIFAR-10とは？
# 飛行機、自動車、鳥、猫、鹿、犬、カエル、馬、船、トラックの画像セット
# 60,000枚の 32x32 のカラー画像で構成され、10種類のラベルが付いている
# 各クラスには6,000枚の画像があり、50,000枚がトレーニング用、10,000枚がテスト用に分かれている

# 画像データの前処理を設定
transform = transforms.Compose([

    # PIL画像やNumPy配列をPyTorchのTensor に変換
    # 画素数の範囲を[0, 255] → [0, 1]に変換 
    transforms.ToTensor(),
    
    # 画像の各チャンネル(R/G/B)の平均(第1引数 mean)と分散(第2引数 std)を指定して正規化する
    # 入力データxに対して、正規化後のデータをx'は以下のようになる
    #           x' = (x - mean) / std
    # 入力範囲[0, 1]を正規化すると、[-1, 1]となる
    # モデルが0を中心に偏りなく学習できるようにするのが正規化の目的
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
# 学習データの設定
# 第1引数: データの保存先
# 第2引数: 学習データか？ → 5万枚の学習用データをセット
# 第3引数: データがない時DLするか？
# 第4引数: 前処理パイプラインの指定
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# データの読み込み方を指定する
# 第1引数: 対象データ
# 第2引数: 1度に読み込むデータの数
# 第3引数: 1エピックごとに読み込む順番をシャッフルするか？（過学習防止）
# 第4引数: 並列プロセスの数
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

# テストデータの設定（学習データの設定のときと引数の値を変えただけ）
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)


# =======================
# 3. クラスラベルの定義（10クラス）
# =======================
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# =======================
# 4. シンプルなCNNモデル定義
# =======================
# PyTorchのニューラルネットモデル定義の基本構文
# nn.Moduleを継承してクラスを作成すると、PyTorchが内部的に計算グラフを構築してくれる
class SimpleCNN(nn.Module):
    
    # モデルの各層を定義する
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Conv2D: 画像からエッジや色を抽出
        # 特徴量の抽出方法を定義
        # 第1引数: 入力チャンネル数
        # 第2引数: 出力チャンネル数
        # 第3引数: カーネルサイズ（フィルターのサイズ）
        # 第4引数: 橋の情報が消えない様に余白を入れる
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        # 特徴量を減らす処理を定義(必要な特徴だけを抽出し計算量を減らす) 
        # 例）次の特徴量の行列にself.poolを適用する
        # [1 2 3 4]
        # [5 6 7 8]
        # [9 A B C]
        # [D E F G]
        # これを2x2のカーネルを2ピクセルごとで分割して、各ステップの最大量を採用する
        # [[1 2], [5 6]] → 6
        # [[3 4], [7 8]] → 8
        # [[9 A], [D E]] → E
        # [[B C], [F G]] → G
        # すると、4x4の特徴量の行列は以下の2x2行列に次元を下げることができる
        # [6 8]
        # [E G]
        self.pool = nn.MaxPool2d(2, 2)

        # 特徴量抽出方法の定義Part2(より深い学習のため)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # 線形変換の処理を定義
        # 1次元配列を64*8*8 → 128ノードに変換
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # 128 → 10ノード（CIFAR-10のクラス数）に変換
        self.fc2 = nn.Linear(128, 10)

    # モデルが実際にデータをどう処理するかを定義
    def forward(self, x):
        
        # ReLU: 特徴量に非線形性を追加
        # 0未満は切り捨て、0以上はそのまま
        # Conv1 → ReLU → Pool
        x = self.pool(torch.relu(self.conv1(x)))

        # Conv2 → ReLU → Pool
        x = self.pool(torch.relu(self.conv2(x)))

        # 抽出した特徴量を1次元配列に変換
        x = x.view(-1, 64 * 8 * 8)
        
        # fc1 → ReLU
        x = torch.relu(self.fc1(x)) 
        
        # 10クラスのいずれかに分類
        x = self.fc2(x)
        return x

# 定義したモデルを生成
model = SimpleCNN().to(device)

# =======================
# 5. 損失関数と最適化アルゴリズム
# =======================
# TIPS: 損失関数(criterion)とは？
# 予測結果のズレを数値で表す関数
# 正解と予測のずれを数値化してモデルにどれくらい間違えたかを伝える
criterion = nn.CrossEntropyLoss()

# TIPS: CrossEntropy(交差エントロピー)とは
# 

# TIPS: 最適化アルゴリズム (optimizer)とは？
# パラメータ（重み）を更新して、損失を減らす
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =======================
# 6. 学習ループ
# =======================
if __name__ == "__main__":
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()               # 勾配初期化
            outputs = model(inputs)             # 順伝播
            loss = criterion(outputs, labels)   # 損失計算
            loss.backward()                     # 逆伝播
            optimizer.step()                    # パラメータ更新

            running_loss += loss.item()
            if i % 100 == 99:    # 100ミニバッチごとに表示
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("✅ 学習完了！")

# =======================
# 7. モデルの保存
# =======================
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cifar10_cnn.pt")
    print("💾 モデル保存完了：models/cifar10_cnn.pt")
