from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import operator
from functools import reduce
from util.util1 import read_data
data = sio.loadmat('./data/cluster_result/result.mat')
H = data['H']
gt = data['gt']
gt = np.squeeze(gt) # np.squeeze（）函数可以删除数组形状中的单维度条目，即把shape中为1的维度去掉，但是对非单维的维度不起作用

X_tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000,random_state=1).fit_transform(H)

plt.figure()
plt.title('epoch=0, acc=0.70')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=gt, s=10, alpha=0.5)
plt.colorbar() # 创建颜色条
plt.show()



trainData, view_num = read_data('./data/handwritten0.mat', 1)  # 将6个view的数据读入并标准化
print(trainData.data[str(5)].shape)

def T_SNE_plot(H,trainData):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    x_tsne = TSNE(n_components=2).fit_transform(H)
    print(x_tsne.shape)
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(1, 1, 1)
    X_1 = x_tsne[:, 0]
    X_2 = x_tsne[:, 1]
    for i in range(x_tsne.shape[0]):
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#F0F8FF', '#FAEBD7']
        for j in range(10):
            if trainData.labels[i] == j:
                plt.scatter(X_1[i], X_2[i], c=color[j])
    plt.show()
    print('success')
T_SNE_plot(trainData.data[str(2)],trainData)