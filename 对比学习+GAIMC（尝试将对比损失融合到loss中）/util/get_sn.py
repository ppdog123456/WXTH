import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def get_sn(view_num, alldata_len, missing_rate): # 随机生成缺失数据矩阵
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """
    one_rate = 1-missing_rate
    if one_rate <= (1 / view_num): # 当缺失率太大时，保证每个sample都能留下一个view的数据
        enc = OneHotEncoder()
        # 随机生成alldata_len行一列的0-5的随机数，然后进行onehot编码
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve  # 2000行6列，每一行5个0一个1
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix # 全是1的2000行6列矩阵
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        # print(f'one_num={one_num}')
        ratio = one_num / (view_num * alldata_len)
        # print(f'ratio={ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        # print(f'matrix_iter=\n{matrix_iter}')
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        # print(f'a={a}')
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix


def save_Sn(Sn, str_name):
    np.savetxt(str_name + '.csv', Sn, delimiter=',')


def load_Sn(str_name):
    return np.loadtxt(str_name + '.csv', delimiter=',')
