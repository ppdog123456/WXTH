from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import  accuracy_score
from . import metrics
import numpy as np
# from kmeans_pytorch import kmeans as KMeans
import torch
def purity(y_true, y_pred):
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # print('调试纯度')
    # print(y_true.shape)
    # print(y_pred.shape)
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):  # 将原本不规范的真实值标签，转化为以0：K为标签的列表。
        y_true[y_true == labels[k]] = ordered_labels[k]  ## 如将[20,24,24,30] -> [0,1,1,2]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)  # 将labels和[np.max(labels)+1]按第一纬度粘合在一起

    for cluster in np.unique(y_pred):  # 由于y_true和y_pred对应的标签不同，所以我们将y_pred中的类别转化为y_true对应位置中出现次数最多的类别
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)  # 将数据分为bins组，统计每一组的个数
        winner = np.argmax(hist)  # axis参数不出现时，此时将数组平铺，找出其中最大的那个值的索引
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def cluster(n_clusters, features, labels, count=10):
    """
    :param n_clusters: number of categories
    :param features: input to be clustered（要聚类的输入）
    :param labels: ground truth of input
    :param count:  times of clustering（聚类次数）
    :return: average acc and its standard deviation,
             average nmi and its standard deviation
    """
    pred_all = []
    # print(type(features))
    # features = torch.from_numpy(features)
    # print(type(features))

    for i in range(count):
        # km = SpectralClustering(n_clusters=n_clusters,n_init=10,affinity='nearest_neighbors',random_state=i).fit(features) # 采用谱聚类会使结果明显变化
        # pred = km.labels_
        km = KMeans(n_clusters=n_clusters,n_init=100)
        pred = km.fit_predict(features)
        pred_all.append(pred)
        # print(f'第{i}次Kmeans')
    gt = np.reshape(labels, np.shape(pred))
    if np.min(gt) == 1:
        gt -= 1
    # print(len(pred_all))
    # print(len(gt))
    acc_avg, acc_std = get_avg_acc(gt, pred_all, count)
    nmi_avg, nmi_std = get_avg_nmi(gt, pred_all, count)
    ri_avg, ri_std = get_avg_RI(gt, pred_all, count)
    f1_avg, f1_std = get_avg_f1(gt, pred_all, count)

    pur = purity(gt, pred)
    return acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std,pur


def get_avg_acc(y_true, y_pred, count):
    acc_array = np.zeros(count)
    for i in range(count):
        acc_array[i] = metrics.acc(y_true, y_pred[i])
    print(f'10次聚类的准确率列表：{acc_array}')
    acc_avg = acc_array.mean()
    acc_std = acc_array.std()
    return acc_avg, acc_std


def get_avg_nmi(y_true, y_pred, count):
    nmi_array = np.zeros(count)
    for i in range(count):
        nmi_array[i] = metrics.nmi(y_true, y_pred[i])
    nmi_avg = nmi_array.mean()
    nmi_std = nmi_array.std()
    return nmi_avg, nmi_std


def get_avg_RI(y_true, y_pred, count):
    RI_array = np.zeros(count)
    for i in range(count):
        RI_array[i] = metrics.rand_index_score(y_true, y_pred[i])
    RI_avg = RI_array.mean()
    RI_std = RI_array.std()
    return RI_avg, RI_std


def get_avg_f1(y_true, y_pred, count):
    f1_array = np.zeros(count)
    for i in range(count):
        f1_array[i] = metrics.f_score(y_true, y_pred[i])
    f1_avg = f1_array.mean()
    f1_std = f1_array.std()
    return f1_avg, f1_std


def get_acc(y_true, y_pred):
    if np.min(y_true) == 1:
        y_true -= 1
    acc_array = metrics.acc(y_true, y_pred)
    return acc_array


def get_nmi(y_true, y_pred):
    if np.min(y_true) == 1:
        y_true -= 1
    acc_array = metrics.nmi(y_true, y_pred)
    return acc_array
