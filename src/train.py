import torch as t
import torch.optim as optim
import torch.functional as f
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

from model import DiffusionModel

device = "cpu"#t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
time_steps = 20
max_beta = 0.7
model = DiffusionModel(time_steps=time_steps, max_beta=max_beta, input_dim=2, hidden1_dim=100, hidden2_dim=50, output_dim=2, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# epoch
epochs = 2000

def train_model():

    # 可視化するための配列宣言
    loss_list = []
    epoch_list = []

    for epoch in range(epochs):
        for batch in dataloader:
            x0 = batch[0]  # (batch_size,2)
            x0 = x0.to(device)

            # 時間（ノイズに関与）をサンプリング
            time = model.sample_time(batch_size=x0.size(0)).to(device)

            # x0とtimeからx_tを生成
            xt = model.noising(x=x0, time=time)
            xt = xt.to(device)

            # ここからノイズを予測
            pred_noise = model(xt, time).to(device)

            # 真値を計算
            x0_pred = x0 - pred_noise

            # ほんとのノイズを持ってくる
            true_noise = model.return_true_noise().to(device)

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
    model_scripted.save(f'/home/onishi/venv/diffusion_model/diffusion-model/models/betaSigmoid{max_beta}_{time_steps}steps_{epochs}epochs.pth')

