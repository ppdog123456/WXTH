
import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf.disable_v2_behavior()

from sklearn.cluster import KMeans
from util.util1 import xavier_init


from tqdm import tqdm
import time
import numpy as np
np.set_printoptions(threshold=np.inf) # 不省略输出的变量
import torch

from util.print_result import print_result





class GAIMCNets:
    """build model
    """

    def __init__(self, view_num, trainLen, labels,layer_size, lsd_dim=128, learning_rate=None, lamb1=1, lamb2=1,lamb3=1,lambc=1,
                 alpha=1,
                 n_cluster=10,):
        """
        :param learning_rate:learning rate of network and h # h是什么？？？
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        """
        # initialize parameter
        if learning_rate is None:
            learning_rate = [0.001, 0.001]
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.batch_size = 256
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lamb3 = lamb3
        self.lambc = lambc
        self.alpha = alpha # （18式）的那个常数
        self.n_cluster = n_cluster
        self.dim_filter = lsd_dim
        self.mask = self.mask_correlated_samples(2*self.batch_size)
        self.labels = labels
        self.trainLen = trainLen
        # initialize latent space data
        self.h, self.h_update = self.H_init('train')  # h和h_updata都是2000*128
        tf.compat.v1.disable_eager_execution()
        self.h_index = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name='h_index')  # 相当于定义了一个变量，提前分配了需要的内存
        self.h_temp = tf.gather_nd(self.h, self.h_index) # 其主要功能是self.h_index根据描述的索引，提取self.h上的元素， 重新构建一个tensor
        # initialize the input data
        self.input = dict()
        self.sn = dict()
        for v_num in range(self.view_num):
            self.input[str(v_num)] = tf.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]],
                                                    name='input' + str(v_num))
            self.sn[str(v_num)] = tf.placeholder(tf.float32, shape=[None, 1], name='sn' + str(v_num))
        print(f'input = {self.input}')
        # ground truth
        # self.gt = tf.placeholder(tf.int32, shape=[None], name='gt')

        # p,q
        self.kmeans = KMeans(n_clusters=self.n_cluster, n_init=100)
        self.mu = tf.Variable(tf.random_normal(shape=(self.n_cluster, self.lsd_dim), name="mu")) # 生成服从正态分布的（10*128）的矩阵，mu也是一个可更新的变量,10个类别的聚类中心
        # assign_mu_op = self.get_assign_cluster_centers_op(self.h_update[0])
        # _ = self.sess.run(assign_mu_op)
        self.q = self._soft_assignment(self.h_update[0], self.mu) # （18式）self.h_update[0]就是综合表示！！！，self.mu是聚类中心

        # print(f'q = {self.q}')
        # self.p = tf.Variable(tf.zeros(shape=(self.trainLen, self.n_cluster)))
        self.p = self.target_distribution(self.q) # 对应（19式）
        # print(f'p = {self.p}')
        # bulid the model
        self.train_op, self.loss = self.bulid_model(self.h_update, self.mu, learning_rate) # 训练网络的优化器和损失函数

        # open session（动态申请显存）
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer()) # 全局变量初始化
        # writer = tf.summary.FileWriter('logs/', self.sess.graph)

    def mask_correlated_samples(self,N):
        mask = tf.ones((N, N))
        # print(mask)
        mask = mask - tf.matrix_diag(tf.diag_part(mask))

        mask = tf.Variable(mask)

        # print(mask)

        for i in range(N//2):
            mask[i, N//2 + i].assign(0)
            mask[N//2 + i, i].assign(0)
        mask = tf.cast(mask, tf.bool) # 将0和1构成的矩阵，转化为由True和False构成的布尔矩阵
        return mask

    def forward_feature(self,h_i, h_j, temperature_f=0.5):
        N = 2 * self.batch_size
        # print(h_i[0])
        # print(h_j)
        h = tf.concat((h_i, h_j), axis=0)  # 将h_i和h_j按行拼接
        # print(f'h = {h}')
        sim = tf.matmul(h, tf.transpose(h)) / temperature_f
        # print(f'sim:{sim}')
        # sim = nn.CrossEntropyLoss(reduction="sum")
        # print(f'sim = {sim}')
        sim_i_j = tf.linalg.diag_part(sim, k=self.batch_size)  # 从sim的对角线开始向右上数第self.batch_size的斜列
        sim_j_i = tf.linalg.diag_part(sim, k=-self.batch_size)  # 向左下数
        positive_samples = tf.reshape(tf.concat((sim_i_j, sim_j_i), axis=0), [N, 1])


        negative_samples = tf.reshape(tf.boolean_mask(sim, self.mask), [N, -1])  # 将数据重构成N行，列数根据数据个数自动计算
        # print(f"negative_samples:{negative_samples}")
        labels = tf.cast(tf.zeros(N), tf.int64)  # 构造一行N列的0向量，long函数将数字或字符串转换为一个长整型。
        # print(f"labels:{labels}")
        logits = tf.concat((positive_samples, negative_samples), axis=1)
        # print(f"logits:{logits}")
        # loss = torch.nn.CrossEntropyLoss(reduction="sum")(logits, labels)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # print(f'loss = {loss}')
        loss = tf.reduce_sum(loss)
        # print(f'loss = {loss}')
        loss /= N
        # print(f'loss = {loss}')
        return loss

    def bulid_model(self, h_update, mu, learning_rate):
        #################################################
        # autoencoder
        #################################################
        auto_loss = []
        F_encoder_net = dict()
        for v_num in range(self.view_num):
            F_encoder_net[str(v_num)] = self.filter_encoder(self.input[str(v_num)], v_num)
            xr = self.filter_decoder(F_encoder_net[str(v_num)],v_num)
            # print(xr)
            auto_loss.append(tf.reduce_mean(tf.losses.mean_squared_error(self.input[str(v_num)], xr)))
        print(auto_loss)
        auto_loss_all = sum(auto_loss)
        # print(f'F_encoder_net  = {F_encoder_net }')

        #################################################
        # 对比学习
        #################################################
        hs = []
        Zs = []
        for v_num in range(self.view_num):
            zs = F_encoder_net[str(v_num)]
            # print(f'第{v_num}个view的zs={zs}')
            h = tf.nn.l2_normalize(self.filter(zs), dim=1)
            # h = self.filter(zs)
            # print(f'h  =  {h}')
            hs.append(h)
            Zs.append(zs)
            # print(len(Zs))
        # print(f"hs = {hs}")
        loss_list = []
        for v in range(self.view_num):
            for w in range(v + 1, self.view_num):
                # print(f'hs[v] = {hs[v]}')
                # print(f'hs[w] = {hs[w]}')
                loss_list.append(self.forward_feature(hs[v], hs[w]))
        con_loss = tf.reduce_sum(loss_list)


        # initialize network（初始化网络）
        global D_loss, G_loss
        net = dict()
        for v_num in range(self.view_num):
            net[str(v_num)] = self.Encoding_net(self.h_temp, v_num)
        print(f'net = {net}')

        #################################################
        # CycleGAN
        #################################################

        cycle_net_1 = dict()
        cycle_net_2 = dict()
        num_v = list(np.arange(self.view_num))
        cycle_Loss_list = []
        for v in num_v:
            num_w = list(np.arange(self.view_num))
            cycle_net_1[str(v)] = self.Encoding_net(hs[v], v)
            num_w.remove(v)
            for w in num_w:
                cycle_net_2[str(v)+str(w)] = self.Encoding_net(cycle_net_1[str(v)], w)
                Loss_vw = tf2.keras.losses.MeanAbsoluteError()(cycle_net_2[str(v)+str(w)],hs[v])
                cycle_Loss_list.append(Loss_vw)
        cycle_Loss = sum(cycle_Loss_list)
        # print(f'cycle_net_2   =   {cycle_net_2} \n 长度为{len(cycle_net_2)}')
        # print(f'cycle_Loss    =   {cycle_Loss}')


        # D
        D_real = {}
        D_logit_real = {}
        D_fake = {}
        D_logit_fake = {}
        D_loss_real = {}
        D_loss_fake = {}

        for v_num in range(self.view_num):
            D_real[v_num], D_logit_real[v_num],weight_D_b0 = self.discriminator(hs[v_num], v_num) # 生成器和鉴别器共用一套参数吗？？不是的
            D_fake[v_num], D_logit_fake[v_num],weight_D_b0 = self.discriminator(net[str(v_num)], v_num)

        # calculate reconstruction loss
        reco_loss = self.reconstruction_loss(net, hs) # 计算重构损失
        # calculate GAN loss （计算GAN损失）

        for v_num in range(self.view_num):
            # 强化鉴别器
            D_loss_real[v_num] = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real[v_num], labels=tf.ones_like( # 对W * X得到的值（D_logit_real[v_num]）进行sigmoid激活，保证取值在0到1之间，然后放在交叉熵的函数中计算Loss
                    D_logit_real[v_num])))
            D_loss_fake[v_num] = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake[v_num], labels=tf.zeros_like(
                    D_logit_fake[v_num])))
            D_loss = D_loss_real[v_num] + D_loss_fake[v_num]
            # 强化生成器（即：net中的参数）
            G_loss = tf.reduce_mean(  # 生成损失
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake[v_num], labels=tf.ones_like(
                    D_logit_fake[v_num])))

        kl_loss = self._kl_divergence(self.q, self.p) # （20式）
        # print(f'con_loss = {con_loss}')
        # print(f'kl_loss = {kl_loss}')

        # all_loss = tf.add(tf.add(tf.add(reco_loss, self.lamb1 * (D_loss + G_loss + self.lambc*cycle_Loss )), self.lamb2 * kl_loss),self.lamb3 * con_loss) # （15式） # 对比损失也会影响最后的h
        all_loss = tf.add(tf.add(reco_loss, self.lamb1 * (D_loss + G_loss + self.lambc*cycle_Loss)), self.lamb2 * kl_loss)
        train_auto = tf.train.AdamOptimizer(0.00001).minimize(auto_loss_all,var_list=[tf.get_collection('weight_FE'),tf.get_collection('weight_FD')])
        train_enc_filter = tf.train.AdamOptimizer(0.00001).minimize(con_loss,var_list=[tf.get_collection('weight_FE'),tf.get_collection('weight_F')])

        Cycle_solver = tf.train.AdamOptimizer(0.00001).minimize(cycle_Loss,var_list=tf.get_collection('weight'))

        D_solver = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list=tf.get_collection('weight_D')) # tf.get_collection() 主要作用：从一个集合中取出变量，优化鉴别损失，只更新鉴别器参数
        G_solver = tf.train.AdamOptimizer(0.00001).minimize(G_loss, var_list=tf.get_collection('weight')) # 优化生成损失，只更新生成器（net）参数
        # train net operator
        # train the network to minimize loss
        train_net_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(reco_loss, var_list=tf.get_collection('weight'))
        # 更新聚类中心
        train_mu_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(kl_loss, var_list=mu) # 通过更新聚类中心mu，最小化kl_loss

        train_hn_op = tf.train.AdamOptimizer(learning_rate[1]) \
            .minimize(all_loss, var_list=h_update) # 通过更新h，最小化all_loss
        return [train_net_op, train_mu_op, train_hn_op, D_solver, G_solver,train_enc_filter,Cycle_solver,train_auto], [reco_loss, kl_loss, all_loss, D_loss,
                                                                              G_loss,con_loss,hs,F_encoder_net,self.input,cycle_Loss,auto_loss_all,xr,Zs]

    def H_init(self, a):
        with tf.compat.v1.variable_scope('H' + a): # 指定了卷积层作用域为H+a
            if a == 'train':
                h = tf.Variable(xavier_init(self.trainLen, self.lsd_dim)) # 创建变量h，2000*128
            h_update = tf.compat.v1.trainable_variables(scope='H' + a) # 更新变量
            # print(f'h = \n{h}')
            # print(f'h_update = \n{h_update}')
        return h, h_update


    def filter_encoder(self,x,v): # 编码器
        weight = self.initialize_weight_F_ENCODER(self.layer_size[v])
        # print(f'weight = {weight}')
        # print(f'Encoding_net中的 h =\n {h}')
        layer = tf.nn.relu((tf.matmul(x, weight['w0']) + weight['b0']))
        layer = tf.nn.relu((tf.matmul(layer, weight['w0_1']) + weight['b0_1']))
        # print(f'filter_encoder = {weight}')
        layer = tf.nn.relu((tf.matmul(layer, weight['w0_2']) + weight['b0_2']))
        # print(layer)
        for num in range(1, len(self.layer_size[v])):
            # print(f'num = {num}')
            layer = tf.matmul(layer, weight['w1']) + weight['b1']
        # print(layer)
        return layer

    def initialize_weight_F_ENCODER(self, dims_net): # 编码器权重
        weight_FE = dict()
        with tf.variable_scope('weight_FE',reuse=tf.AUTO_REUSE):
            weight_FE['w0'] = tf.Variable(tf2.keras.initializers.HeUniform()(shape=(dims_net[1], 500)))  # 采用何凯明的初始化
            weight_FE['b0'] = tf.Variable(tf.random.uniform(shape=(1,500)))
            tf.add_to_collection("weight_FE", weight_FE['w' + str(0)]) # 将元素weight_D['w' + str(0)]添加到weight_D中
            tf.add_to_collection("weight_FE", weight_FE['b' + str(0)])
            weight_FE['w0_1'] = tf.Variable(tf2.keras.initializers.HeUniform()(shape=(500, 500)))
            weight_FE['b0_1'] = tf.Variable(tf.random.uniform(shape=(1,500)))
            tf.add_to_collection("weight_FE", weight_FE['w0_' + str(1)])
            tf.add_to_collection("weight_FE", weight_FE['b0_' + str(1)])
            weight_FE['w0_2'] = tf.Variable(tf2.keras.initializers.HeUniform()(shape=(500, 2000)))
            weight_FE['b0_2'] = tf.Variable(tf.random.uniform(shape=(1,2000)))
            tf.add_to_collection("weight_FE", weight_FE['w0_' + str(2)])
            tf.add_to_collection("weight_FE", weight_FE['b0_' + str(2)])
            for num in range(1, len(dims_net)):
                weight_FE['w' + str(num)] = tf.Variable(tf2.keras.initializers.HeUniform()(shape=(2000, 512)))
                weight_FE['b' + str(num)] = tf.Variable(tf.random.uniform(shape=(1,512)))
                tf.add_to_collection("weight_FE", weight_FE['w' + str(num)])
                tf.add_to_collection("weight_FE", weight_FE['b' + str(num)])
        return weight_FE # #

    def filter_decoder(self,x,v): # 解码器
        weight = self.initialize_weight_F_DNCODER(self.layer_size[v])
        # print(f'weight = {weight}')
        # print(f'Encoding_net中的 h =\n {h}')
        layer = tf.nn.relu((tf.matmul(x, weight['w0']) + weight['b0']))
        layer = tf.nn.relu((tf.matmul(layer, weight['w0_1']) + weight['b0_1']))
        # print(f'filter_encoder = {weight}')
        layer = tf.nn.relu((tf.matmul(layer, weight['w0_2']) + weight['b0_2']))
        # print(layer)
        for num in range(1, len(self.layer_size[v])):
            # print(f'num = {num}')
            layer = tf.matmul(layer, weight['w1']) + weight['b1']
        # print(layer)
        return layer

    def initialize_weight_F_DNCODER(self, dims_net): # 解码器权重
        weight_FD = dict()
        with tf.variable_scope('weight_FD',reuse=tf.AUTO_REUSE):
            weight_FD['w0'] = tf.Variable(tf2.keras.initializers.HeUniform()(shape=(512, 2000)))  # 采用何凯明的初始化
            weight_FD['b0'] = tf.Variable(tf.random.uniform(shape=(1,2000)))
            tf.add_to_collection("weight_FD", weight_FD['w' + str(0)]) # 将元素weight_D['w' + str(0)]添加到weight_D中
            tf.add_to_collection("weight_FD", weight_FD['b' + str(0)])
            weight_FD['w0_1'] = tf.Variable(tf2.keras.initializers.HeUniform()(shape=(2000, 500)))
            weight_FD['b0_1'] = tf.Variable(tf.random.uniform(shape=(1,500)))
            tf.add_to_collection("weight_FD", weight_FD['w0_' + str(1)])
            tf.add_to_collection("weight_FD", weight_FD['b0_' + str(1)])
            weight_FD['w0_2'] = tf.Variable(tf2.keras.initializers.HeUniform()(shape=(500, 500)))
            weight_FD['b0_2'] = tf.Variable(tf.random.uniform(shape=(1,500)))
            tf.add_to_collection("weight_FD", weight_FD['w0_' + str(2)])
            tf.add_to_collection("weight_FD", weight_FD['b0_' + str(2)])
            for num in range(1, len(dims_net)):
                weight_FD['w' + str(num)] = tf.Variable(tf2.keras.initializers.HeUniform()(shape=(500, dims_net[1])))
                weight_FD['b' + str(num)] = tf.Variable(tf.random.uniform(shape=(1,dims_net[1])))
                tf.add_to_collection("weight_FD", weight_FD['w' + str(num)])
                tf.add_to_collection("weight_FD", weight_FD['b' + str(num)])
        return weight_FD # #

    def filter(self,z): # 对比学习过滤器
        weight = self.initialize_weight_F()
        # layer = tf.linalg.normalize ((tf.matmul(z, weight['w0']) + weight['b0']),axis=0)
        layer = (tf.matmul(z, weight['w0']) + weight['b0'])
        return layer


    def initialize_weight_F(self): # 对比学习过滤器权重
        weight_F = dict()
        with tf.variable_scope('weight_F',reuse=tf.AUTO_REUSE):
            weight_F['w0'] = tf.get_variable('w0', initializer=tf2.keras.initializers.HeUniform()(shape=(512, self.dim_filter)))
            weight_F['b0'] = tf.get_variable('b0', initializer=tf.zeros(shape=(1,self.dim_filter)))
            tf.add_to_collection("weight_F", weight_F['w' + str(0)]) # 将元素weight_D['w' + str(0)]添加到weight_D中
            tf.add_to_collection("weight_F", weight_F['b' + str(0)])
        return weight_F

    def discriminator(self, x, v):  # 判别器输入原数据
        weight = self.initialize_weight_D(self.layer_size[v], view_D=v)
        # print(f'weight_D = {weight}')
        D_h1 = tf.nn.relu(tf.matmul(x, weight['w0']) + weight['b0'])  # 先线性计算，再激活
        D_logit = tf.matmul(D_h1, weight['w1']) + weight['b1']  # 再线性计算D的分对数
        # D_logit = D_h1
        D_prob = tf.nn.sigmoid(D_logit)  # 将线性结果概率化（线性结果乘sigmoid）

        return D_prob, D_logit, weight['b0']

    def initialize_weight_D(self, dims_net, view_D):  # 初始化权重weight_D（判别器用的权重）
        weight_D = dict()
        with tf.variable_scope('weight_D' + str(view_D), reuse=tf.AUTO_REUSE):
            weight_D['w0'] = tf.get_variable('w0', initializer=xavier_init(self.lsd_dim, dims_net[0]))
            # variable_summaries(weight_D['w0'])
            weight_D['b0'] = tf.get_variable('b0', initializer=tf.zeros([dims_net[0]]))
            # variable_summaries(weight_D['b0'])
            tf.add_to_collection("weight_D", weight_D['w' + str(0)])  # 将元素weight_D['w' + str(0)]添加到weight_D中
            tf.add_to_collection("weight_D", weight_D['b' + str(0)])
            for num in range(1, len(dims_net)):
                weight_D['w' + str(num)] = tf.get_variable('w1', initializer=xavier_init(dims_net[num - 1], 1))
                weight_D['b' + str(num)] = tf.get_variable('b1', initializer=tf.zeros([1]))
                tf.add_to_collection("weight_D", weight_D['w' + str(num)])
                tf.add_to_collection("weight_D", weight_D['b' + str(num)])
        return weight_D

    def Encoding_net(self, h, v): # 生成器，128维的h先和W0（128*150维）线性相乘，然后再得到的150维的数据再和Wi相乘，输出结果为原数据的维度的向量
        weight = self.initialize_weight(self.layer_size[v],v)
        # print(f'weight = {weight}')
        # print(f'Encoding_net中的 h =\n {h}')
        layer = tf.tanh(tf.matmul(h, weight['w0']) + weight['b0'])
        # print(layer)
        layer = tf.tanh(tf.matmul(layer, weight['w0_1']) + weight['b0_1'])
        for num in range(1, len(self.layer_size[v])):
            # print(f'num = {num}')
            layer = tf.nn.dropout(tf.matmul(layer, weight['w' + str(num)]) + weight['b' + str(num)], 0.9) # 随机断开0.1的连接
        # print(layer)
        return layer

    def initialize_weight(self, dims_net,view): # 初始化权重weight（编码器用的权重）
        all_weight = dict()
        with tf.variable_scope('weight'+ str(view), reuse=tf.AUTO_REUSE):
            all_weight['w0'] = tf.get_variable('w0',initializer=xavier_init(self.lsd_dim,200))
            all_weight['b0'] = tf.get_variable('b0',initializer=tf.zeros([200]))
            tf.add_to_collection("weight", all_weight['w' + str(0)]) # 将元素all_weight['w' + str(0)]添加到列表weight中
            tf.add_to_collection("weight", all_weight['b' + str(0)])
            all_weight['w0_1'] = tf.get_variable('w0_1',initializer=xavier_init(200, 500))
            all_weight['b0_1'] = tf.get_variable('b0_1',initializer=tf.zeros([500]))
            tf.add_to_collection("weight", all_weight['w0_' + str(1)])  # 将元素all_weight['w' + str(0)]添加到列表weight中
            tf.add_to_collection("weight", all_weight['b0_' + str(1)])
            for num in range(1, len(dims_net)):
                all_weight['w' + str(num)] = tf.get_variable('w1',initializer=xavier_init(500, self.dim_filter))
                all_weight['b' + str(num)] = tf.get_variable('b1',initializer=tf.zeros([self.dim_filter]))
                tf.add_to_collection("weight", all_weight['w' + str(num)])
                tf.add_to_collection("weight", all_weight['b' + str(num)])
        return all_weight

    def reconstruction_loss(self, net,H): # 计算重构损失，对应第(16)式
        loss = 0
        for num in range(self.view_num):
            loss = loss + tf.reduce_sum(
                tf.pow(tf.subtract(net[str(num)], H[num])   # tf.subtract(a,b)矩阵a，b对应元素相减
                       , 2.0) * self.sn[str(num)] # 其中self.sn[str(num)]就是缺失矩阵
            )
        return loss

    def get_assign_cluster_centers_op(self, features): # 得到指派聚类中心，features为聚类中心个数
        # init mu
        print("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        print("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_) # 将self.mu的值变为kmeans.cluster_centers_
                                                            ## (kmeans.cluster_centers_查看聚类中心坐标)

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.
            (实现了一个软赋值作为分配样本i到集群的概率)
        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """

        def _pairwise_euclidean_distance(a, b):  # 成对欧几里得距离
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.trainLen, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True)) #
            return res
        # print(f'h_update[0] = {embeddings}')
        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)  #
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q   # （18式）

    def target_distribution(self, q): #
        l = tf.reduce_sum(q, axis=0) #
        p = q ** 2 / l
        k = tf.reduce_sum(p, axis=0)
        p = p / k
        return p

    def _kl_divergence(self, target, pred): # 对应（20式）
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))

    def train(self, data, sn, epoch, step=None):
        

        index = np.array([x for x in range(self.trainLen)])

        sn = sn[index]
        feed_dict = {self.input[str(v_num)]: data[str(v_num)][index] for v_num in range(self.view_num)} # feed_dict的作用是给使用placeholder创建出来的tensor赋值
        # print(f'feed_dict = {feed_dict[self.input[str(0)]].shape}')
        feed_dict.update(
            {self.sn[str(i)]: sn[:, i].reshape(self.trainLen, 1) for i in range(self.view_num)})  # 这个updata从何而来啊？
        # feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})
        Auto_loss_1 = 0
        for i in range(step[3]):

            _,Auto_loss,xrf = self.sess.run(
                [self.train_op[7], self.loss[10],self.loss[11]], feed_dict=feed_dict)

            if abs(Auto_loss_1-Auto_loss) < 0.0001:
                break
            else:
                Auto_loss_1 = Auto_loss


        batch_size = 256
        n_batch = feed_dict[self.input[str(0)]].shape[0]//batch_size

        for k in range(step[2]):
            for batch in range(n_batch):


                con_feed_dict = {self.input[str(v_num)]:feed_dict[self.input[str(v_num)]][batch*batch_size:(batch+1)*batch_size] for v_num in range(self.view_num)}

                _, con_LOSS, low_feature = self.sess.run([self.train_op[5], self.loss[5],  self.loss[7]],feed_dict=con_feed_dict)





        con_loss_list = []
        All_loss_1 = 0
        for iter in tqdm(range(epoch)):
            # update the network

            _, Reconstruction_LOSS, D_loss_curr, G_loss_curr,C_loss_curr = self.sess.run(
                [self.train_op[6], self.loss[0], self.loss[3], self.loss[4],self.loss[9]] ,feed_dict=feed_dict)

            for j in range(step[0]):
                # 重构损失
                # print(j)
                _, Reconstruction_LOSS, D_loss_curr, G_loss_curr = self.sess.run(
                    [self.train_op[0], self.loss[0], self.loss[3], self.loss[4]], feed_dict=feed_dict)

                _, Reconstruction_LOSS, D_loss_curr, G_loss_curr = self.sess.run(
                    [self.train_op[3], self.loss[0], self.loss[3], self.loss[4]], feed_dict=feed_dict)

                _, Reconstruction_LOSS, D_loss_curr, G_loss_curr = self.sess.run(
                    [self.train_op[4], self.loss[0], self.loss[3], self.loss[4]], feed_dict=feed_dict)

            for j in range(step[0]):
                _, Reconstruction_LOSS, KL_LOSS = self.sess.run(
                    [self.train_op[1], self.loss[0], self.loss[1]], feed_dict=feed_dict)


            for j in range(step[1]):  # # 对比损失也会影响最后的综合表示h
                _, Reconstruction_LOSS, All_loss = self.sess.run(
                    [self.train_op[2], self.loss[0], self.loss[2]], feed_dict=feed_dict)
            # print(f'All_loss = {All_loss}')
            if abs(All_loss_1-All_loss) < 0.001:
                break
            else:
                All_loss_1 = All_loss

    def get_h(self):
        lsd = self.sess.run(self.h)
        # H_up = self.sess.run(self.h_update)

        return lsd # lsd为潜在空间数据
