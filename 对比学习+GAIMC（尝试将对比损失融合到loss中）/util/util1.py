import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


class DataSet(object): # 类继承了object对象，拥有了好多可操作对象，这些都是类中的高级特性。

    def __init__(self, data, view_number, labels):
        """
        Construct a DataSet.
        """
        self.data = dict()
        self._num_examples = data[0].shape[0]
        self._labels = labels
        for v_num in range(view_number):
            self.data[str(v_num)] = data[v_num]

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


# def Normalize(data):
#     """
#     :param data:Input data
#     :return:normalized data
#     """
#     m = np.mean(data)
#     mx = np.max(data)
#     mn = np.min(data)
#     return (data - m) / (mx - mn)
def Normalize(data):
    # min_val = np.min(x)
    # max_val = np.max(x)
    # x = (x - min_val) / (max_val - min_val)
    # return x

    scaler = MinMaxScaler([0, 1])
    norm_data = scaler.fit_transform(data)
    return norm_data


def read_data(str_name, Normal=1):
    """
    :param str_name:path and dataname
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    print(str_name)
    data = sio.loadmat(str_name) # 加载matlab文件
    # print(data['X'])
    view_number = data['X'].shape[1]
    X = np.split(data['X'], view_number, axis=1) # 将data['x']按axis=1平均分成view_number份
    # print(X[0][0][0].transpose())
    X_all = []
    if str_name=='./data/MSRC.mat' :
        for v_num in range(view_number):
            X_all.append(X[v_num][0][0]) # 矩阵不转置
        if min(data['Y']) == 0:
            labels = (data['Y'] + 1)
        else:
            labels = data['Y']

    if str_name=='./data/3sources-3view.mat' or str_name=='./data/bbcsport-4view.mat' or  str_name=='./data/ORL_3view.mat' or str_name=='./data/Caltech101-7_6view.mat' or str_name=='./data/BDGP_4view.mat':
        for v_num in range(view_number):
            X_all.append(X[v_num][0][0].transpose()) # 矩阵转置
        if min(data['Y']) == 0:
            labels = (data['Y'] + 1)
        else:
            labels = data['Y']
    else:
        for v_num in range(view_number):
            X_all.append(X[v_num][0][0].transpose()) # transpose()矩阵转置
        if min(data['gt']) == 0:
            labels = (data['gt'] + 1)
        else:
            labels = data['gt']
    if Normal == 1:
        for v_num in range(view_number):
            X_all[v_num] = Normalize(X_all[v_num])
    # print(data['gt'])



    # print(labels)
    if view_number==6:
        X_all[0], X_all[1],X_all[2],X_all[3],X_all[4],X_all[5], labels = shuffle(X_all[0], X_all[1], X_all[2],X_all[3],X_all[4],X_all[5],labels)
    if view_number==5:
        X_all[0], X_all[1], X_all[2], X_all[3], X_all[4], labels = shuffle(X_all[0], X_all[1], X_all[2],
                                                                                     X_all[3], X_all[4],
                                                                                     labels)
    if view_number==4:
        X_all[0], X_all[1], X_all[2], X_all[3], labels = shuffle(X_all[0], X_all[1], X_all[2],
                                                                                     X_all[3],
                                                                                     labels)
    if view_number==3:
        X_all[0], X_all[1],X_all[2], labels = shuffle(X_all[0], X_all[1],X_all[2], labels)
    if view_number==2:
        X_all[0], X_all[1], labels = shuffle(X_all[0], X_all[1], labels)
    # print('----------------------------------------------')
    # print(X_all)
    # print(labels.shape)
    traindata = DataSet(X_all, view_number, np.array(labels))

    return traindata, view_number

'''Xavier初始化在Relu层表现不好，主要原因是relu层会将负数映射到0，影响整体方差。'''
def xavier_init(fan_in, fan_out, constant=1): # 这个函数没看懂（产生fan_in行fan_out列，范围在low-high之间的随机数）
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)
