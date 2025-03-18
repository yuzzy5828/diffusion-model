import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.functional as f

import numpy as np
import matplotlib.pyplot as plt

def sampling(model, n_samples=1000, steps=50, device='cpu'):

    # 完全なノイズつくる
    x = t.randn(n_samples, 2, device=device)
    
    # 逆拡散をsteps分やる
    for time in range(steps, 0, -1):
        # 現在のステップに対応する時間
        time_tensor = t.ones(n_samples, device=device) * time
        
        # モデルによるノイズ予測
        with t.no_grad():
            predicted_noise = model(x, time_tensor)
        
        # 現在のステップに対応するalphaとbeta
        alpha = model.alphas[time - 1]
        alpha_bar = model.alphas_cumprod[time - 1]
        beta = model.betas[time - 1]
        
        # 前のステップのアルファ累積
        if time > 1:
            alpha_bar_prev = model.alphas_cumprod[time - 2]
        else:
            alpha_bar_prev = t.tensor(1.0)
        
        # ノイズ係数
        sigma = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        
        # サンプリング
        if time > 1:
            noise = t.randn_like(x)
        else:
            noise = t.zeros_like(x)  # 最後のステップではノイズを加えない
        
        # 次のステップの潜在変数を計算
        x = (1 / t.sqrt(alpha)) * (x - ((1 - alpha) / t.sqrt(1 - alpha_bar)) * predicted_noise) + t.sqrt(sigma) * noise
    
    return x

def main():
    # modelのインポート
    model_from_script = t.jit.load('/home/yujiro/venv/diffusion_model/models/model.pth', map_location="cpu")
    model_from_script.eval()

    # 逆拡散プロセスでサンプル生成
    x_denoise = sampling(model_from_script, n_samples=1000, steps=50)

    # plt 用にnumpy変換
    x_denoise = x_denoise.cpu().numpy()

    # GMMデータを可視化
    np.random.seed(0)
    n_samples = 1000
    means = np.array([[0, 0], [10, 10], [-10, 10], [10, -10], [-10, -10]])
    covariances = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], 
                            [[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], 
                            [[1, 0.5], [0.5, 1]]])
    weights = np.array([0.2, 0.3, 0.1, 0.2, 0.2]) # 和が１

    X_gmm = np.concatenate([np.random.multivariate_normal(mean, cov, int(weight*n_samples))
                        for mean, cov, weight in zip(means, covariances, weights)])


    X_mean = X_gmm.mean(keepdims=True)
    X_std = X_gmm.std(keepdims=True)
    X_gmm = (X_gmm - X_mean) / (X_std + 1e-7)


    plt.scatter(X_gmm[:,0], X_gmm[:,1], color='red', alpha=0.5, label='GMM')
    # Diffusion 生成結果を可視化する
    plt.scatter(x_denoise[:,0], x_denoise[:,1], color='blue', alpha=0.5, label='Diffusion')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()