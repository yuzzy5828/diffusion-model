import torch as t
import torch.optim as optim
import torch.functional as f
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

from model import DiffusionModel

# まずGMMからランダムにデータを生成する
np.random.seed(0)
n_samples = 1000
means = np.array([[0, 0], [10, 10], [-10, 10], [10, -10], [-10, -10]])
covariances = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], 
                        [[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], 
                        [[1, 0.5], [0.5, 1]]])
weights = np.array([0.2, 0.3, 0.1, 0.2, 0.2]) # 和が１

X_gmm = np.concatenate([np.random.multivariate_normal(mean, cov, int(weight*n_samples))
                    for mean, cov, weight in zip(means, covariances, weights)])

# plt.scatter(X_gmm[:, 0], X_gmm[:, 1])
# plt.show()

# PyTorch Tensorへ
X_gmm = t.from_numpy(X_gmm).float()

X_mean = X_gmm.mean(dim=0, keepdim=True)
X_std = X_gmm.std(dim=0, keepdim=True)
X_gmm = (X_gmm - X_mean) / (X_std + 1e-7)

# DataLoaderへ
dataset = TensorDataset(X_gmm)
dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)

# モデルの設定
model = DiffusionModel(1000, 2, 100, 50, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_model():

    # 可視化するための配列宣言
    loss_list = []
    epoch_list = []

    epochs = 100000

    for epoch in range(epochs):
        for batch in dataloader:
            x0 = batch[0]  # (batch_size,2)

            # 時間（ノイズに関与）をサンプリング
            time = model.sample_time(batch_size=x0.size(0), device=x0.device)

            # x0とtimeからx_tを生成
            xt = model.noising(x=x0, time=time)

            # ここからノイズを予測
            pred_noise = model(xt, time)

            # 真値を計算
            x0_pred = x0 - pred_noise

            # ほんとのノイズを持ってくる
            true_noise = model.return_true_noise()

            # ノイズを比較する方と，真値を比較する方何方も考えてみる
            loss = t.mean((pred_noise - true_noise)**2)
            # loss = t.mean((x0_pred - x0)**2)

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 可視化ようで保存
            # if epoch % 100 == 0:
            loss_list.append(loss.item())
            epoch_list.append(epoch)
    
        print(f"Epoch {epoch}, loss = {loss.item():.4f}")

    plt.plot(epoch_list, loss_list)
    plt.show()

if __name__ == "__main__":
    train_model()
    model_scripted = t.jit.script(model)
    model_scripted.save('/home/yujiro/venv/diffusion_model/models/beta0.05_100steps_100000epochs    .pth')

