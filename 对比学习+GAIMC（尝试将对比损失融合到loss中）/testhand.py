import numpy as np
import scipy.io as sio
from util.util1 import read_data
from util.get_sn import get_sn
from util.model import GAIMCNets
from util.print_result import print_result
import os
import warnings
import torch
import argparse
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


warnings.filterwarnings("ignore")   # 忽略匹配的警告
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# CUB
# handwritten0
# animal
def con_GAICM(lamb1=1,lamb2=1,lamb3=1,lambc=1,Dataname='handwritten0',step=[5,5,50,200],learning_rate = [0.001, 0.01]):

    # print(f'对比学习次数：{con_epoch}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsd-dim', type=int, default=128,
                        help='dimensionality of the latent space data [default: 128]')  # 潜在空间数据的维数
    parser.add_argument('--epochs-train', type=int, default=30, metavar='N',
                        help='number of epochs to train [default: 30]')
    parser.add_argument('--lamb1', type=float, default=lamb1,   # GAN损失超参数
                        help='trade off parameter [default: 1]')
    parser.add_argument('--lamb2', type=float, default=lamb2, # KL损失超参数
                        help='trade off parameter [default: 1]')
    parser.add_argument('--lamb3', type=float, default=lamb3,   # 对比损失超参数
                        help='trade off parameter [default: 1]')
    parser.add_argument('--lambc', type=float, default=lambc,  # 对比损失超参数
                        help='trade off parameter [default: 1]')
    parser.add_argument('--missing-rate', type=float, default=0,
                        help='view missing rate [default: 0]')
    parser.add_argument('--alpha', type=float, default=1,  # 权衡参数，用于（18）式
                        help='trade off parameter [default: 1]')
    args = parser.parse_args()
    # handwritten0
    # read data

    trainData, view_num = read_data('./data/'+str(Dataname)+'.mat', 1)  # 将6个view的数据读入并标准化
    print(trainData.data[str(0)].shape)


    outdim_size = [trainData.data[str(i)].shape[1] for i in range(view_num)]  # 输出数据尺寸
    print(f'输出数据尺寸   =   {outdim_size}')

    n_cluster = len(set(trainData.labels.reshape(trainData.num_examples)))  # 确定类别个数

    layer_size = [[150, outdim_size[i]] for i in range(view_num)]  # 设置每层的尺寸
    print(f'layer_size = {layer_size}')
    # set parameter
    epoch = [args.epochs_train]
    learning_rate = [0.001, 0.01]


    Sn = get_sn(view_num, trainData.num_examples, args.missing_rate)  # 得到缺失数据矩阵
    # print(f'Sn=\n{Sn[1]}')
    Sn_train = Sn[np.arange(trainData.num_examples)]  # Sn和Sn_train一模一样又有什么意义呢？？？

    tf.reset_default_graph() # 用于清除默认图形堆栈并重置全局默认图形。
    model = GAIMCNets(view_num, trainData.num_examples, trainData.labels, layer_size, args.lsd_dim, learning_rate,
                      args.lamb1, args.lamb2,args.lambc,args.alpha, n_cluster)
    model.train(trainData.data, Sn_train, epoch[0],step=step) # step=[GAN+u,h,对比学习]

    H = model.get_h() # H是模型输出的结果2000*128，将H进行K-means聚类才能得到最终的结果

    acc,nmi,ri,f1,pur = print_result(n_cluster, H, trainData.labels)
    acc_mat_lamb[lamb1_list.index(lamb1),lamb2_list.index(lamb2)] = acc

    result = './data/cluster_result/result.mat'
    sio.savemat(result, {'H': H, 'gt': trainData.labels})
    return acc_mat_lamb,acc,nmi,ri,f1,pur
if __name__ == "__main__":

    lamb1_list = [1]
    lamb2_list = [1]
    lamb3_list = [1]
    lambc_list = [1]
    step = [5,1,50,200]
    # learning_rate = [0.001, 0.01] # handwritten0,CUB
    learning_rate = [0.001, 0.01]
    acc_mat_lamb = np.zeros(shape=(len(lamb1_list),len(lamb2_list)))
    Dataname_list = ['handwritten0']

    list_all = []
    # MSRC_acc_list = []
    for con in range(1):
        for Dataname in Dataname_list:

            if Dataname == 'handwritten0':
                step = [5,1,50,200]


            for lamb1 in lamb1_list:
                for lamb2 in lamb2_list:
                    for lambc in lambc_list:
                        for lamb3 in lamb3_list:
                            acc_list_all = []
                            nmi_list_all = []
                            ri_list_all = []
                            f1_list_all = []
                            pur_list_all = []
                            for j in range(1): # 重20次实验，取平均值
                                acc_mat_lamb,acc,nmi,ri,f1,pur = con_GAICM(lamb1=lamb1,lamb2=lamb2,lamb3=lamb3,lambc=lambc,Dataname=Dataname,learning_rate = learning_rate,step=step)
                                # MSRC_acc_list.append([con,acc])
                                acc_list_all.append(acc)
                                nmi_list_all.append(nmi)
                                ri_list_all.append(ri)
                                f1_list_all.append(f1)
                                pur_list_all.append(pur)
                            print(f'重复试验次数为 = {len(acc_list_all)}')
                            print(f'ACC_LIST = {len(acc_list_all)}')
                            print(f'acc = {acc_list_all}')
                            print(f'nmi = {nmi_list_all}')
                            print(f'ri  = {ri_list_all}')
                            print(f'f1  = {f1_list_all}')
                            print(f'pur = {pur_list_all}')
                            print(f'测试cycle损失函数超参数\nlamb1={lamb1}\nlamb2={lamb2}\nlamb3={lamb3}\ndata={Dataname}\nlearning_rate={learning_rate}\n加入cycleGAN后的模型均值：{np.mean(acc_list_all)}   \n加入GAN后的模型标准差：{np.std(acc_list_all)}\n')
                            # con_epoch_list.append(np.mean(acc_list_all))
                            list_all.append([Dataname,lamb1,lamb2,lamb3,lambc,learning_rate,step,np.mean(acc_list_all),np.std(acc_list_all),np.mean(nmi_list_all),np.mean(ri_list_all),np.mean(f1_list_all),np.mean(pur_list_all)])








