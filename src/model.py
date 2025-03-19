import torch as t
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class DiffusionModel(nn.Module):
    def __init__(self, time_steps, max_beta, input_dim, hidden1_dim, hidden2_dim, output_dim, device):
        super().__init__()
        self.device = device
        self.time_steps = time_steps
        
        # ノイズスケジューラーをていぎ
        # betaを線形にスケジュール
        # self.betas = t.linspace(1e-4, max_beta, steps=time_steps).to(device)
        # betaをcosでスケジュール
        self.betas = 0.7 * t.sigmoid(t.linspace(-4, 4, steps=time_steps)).to(device)
        self.alphas = 1 - self.betas
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = t.cumprod(self.alphas, dim=0).to(device)

        self.x_t = []

        # resnetを定義（今はmlpで，+2は時間のembbeding）
        self.resnet = nn.Sequential(
            nn.Linear(input_dim+ 2, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)
        )

    def sample_time(self, batch_size):
        """
        1~self.time_steps の中からランダムに整数を返す例
        """
        #times = t.ones((batch_size, )) * t.randint(1, self.time_steps, (1, 1), device=device).float()
        times = t.randint(1, self.time_steps + 1, (batch_size,), device=self.device).float().to(self.device)
        return times
    
    def noise_scheduling(self, time):
        for tau in range(time):
            self.alpha *= (1 - t.sigmoid(time))
        self.beta = 1 - self.alpha
    
    def noising(self, x, time):
        # 拡散過程．ノイズを入れるパート（バッチサイズごとにtimeを変える）
        alpha_bar_t = self.alphas_cumprod[time.long() - 1]
        alpha_bar_t = alpha_bar_t.view(-1,1).to(self.device)

        self.noise = t.randn_like(x).to(self.device)

        self.x_t = t.sqrt(alpha_bar_t) * x + t.sqrt(1 - alpha_bar_t) * self.noise

        return self.x_t
    
    def forward(self, x_t, time):
        # ここでNNを組み立てる．いっかいUNetベースのアーキテクチャを立ててみる．

        # マックスで
        gamma =  0.5 * t.pi / 180.0 
        # まずtimeに関するembeddingベクトルを作る．今回はデータが2次元だし，時間についても2次元くらいでいいか？
        time = time.view(-1,1)

        time_cos = t.cos(time * gamma)
        time_sin = t.sin(time * gamma)
        time_embed_vector = t.cat([time_cos, time_sin], dim=-1)  # (batch_size,2)

        # ここでは4 × 1のベクトルの形になっていることを期待
        input = t.cat([x_t, time_embed_vector], dim=-1) 

        # このもとでResNetとかで一回推論
        out = self.resnet(input)

        #outputを出力
        return out
    
    def return_true_noise(self):
        return self.noise
