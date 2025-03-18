import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.functional as f

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

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

noise = np.random.normal(1, 0.7, (1000, 2))
X_gmm = X_gmm + noise


plt.scatter(X_gmm[:,0], X_gmm[:,1], color='red', alpha=0.5, label='GMM')
plt.show()