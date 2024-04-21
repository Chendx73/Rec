import scipy.sparse as sp
import numpy as np
from time import time
from collections import defaultdict

import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
from utils.SelfLog import Log

log = Log()


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
        log.logger.info("init dataset")


class Loader(BasicDataset):
    def __init__(self, path, args):
        super().__init__()
        self.args = args
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        # train_unique_users用来记录所有用户唯一的索引或ID
        # train_item用来记录所有用户交互过的itemID,并且顺序对应
        # train_user用来标识train_item中itemID被交互的用户ID
        # train_data_size实际上就等于train_item的总数
        train_unique_users, train_item, train_user = [], [], []
        test_unique_users, test_item, test_user = [], [], []
        self.train_data_size = 0
        self.test_data_size = 0

        user_cnt = 0
        item_cnt = 0
        with open(train_file, encoding='utf8') as f:
            for line in f.readline():
                if len(line) > 0:
                    l = line.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    train_unique_users.append(uid)
                    train_user.extend([uid] * len(items))
                    train_item.extend(items)
                    user_cnt += 1
                    item_cnt = max(item_cnt, max(items) + 1)
                    self.train_data_size += len(items)
        self.n_user = user_cnt
        self.m_item = item_cnt
        self.train_unique_users = np.array(train_unique_users)
        self.train_user = np.array(train_user)
        self.train_item = np.array(train_item)

        with open(test_file, encoding='utf8') as f:
            for line in f.readline():
                if len(line) > 0:
                    l = line.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    test_unique_users.append(uid)
                    test_user.extend([uid] * len(items))
                    test_item.extend(items)
                    user_cnt += 1
                    self.test_data_size += len(items)
        self.test_unique_users = np.array(test_unique_users)
        self.test_user = np.array(test_user)
        self.test_item = np.array(test_item)

        self.Graph = None
        print('{} interactions for training'.format(self.train_data_size))
        print('{} interactions for testing'.format(self.test_data_size))
        print('dataset Sparsity: {}'.format((self.train_data_size + self.test_data_size) / (self.n_user + self.m_item)))
        log.logger.info('{} interactions for training'.format(self.train_data_size))
        log.logger.info('{} interactions for testing'.format(self.test_data_size))
        log.logger.info(
            'dataset Sparsity: {}'.format((self.train_data_size + self.test_data_size) / (self.n_user + self.m_item)))

        # 用户,项目,二部图
        # csr_matrix((数据,(行索引,列索引)),形状)
        self.user_item_net = csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                        shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()  # 从行的维度获取用户的度
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # 记录正样本
        self._allPos = self.get_user_pos_items(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print('dataset is ready')
        log.logger.info('dataset is ready')

    def get_sparse_graph(self):
        print('loading adjacency matrix')
        norm_adj = None
        if self.Graph is None:
            try:
                pass  # 读入训练过的邻接矩阵.npz文件
            except:
                print('generating adjacency matrix')
                s = time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.user_item_net.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()

                # 度矩阵
                row_sum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(row_sum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                # 正则化矩阵
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print('costing {}s, saved norm_mat...'.format(end - s))

            if self.args.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print('done split matrix')
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.args.device)
                print('''don't split the matrix''')
        return self.Graph

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_user + self.m_item) // self.args.folds
        for i_fold in range(self.args.folds):
            start = i_fold * fold_len
            if i_fold == self.args.folds - 1:
                end = self.n_user + self.m_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.args.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __build_test(self) -> defaultdict:
        """
        构造test集合,key为用户ID,value为该用户ID交互过的item列表
        :return:
        """
        test_data = defaultdict(list)
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            test_data[user].append(item)
        return test_data

    def get_user_pos_items(self, users: list) -> list:
        """
        获取用户的正样本item
        :param users: 用户列表
        :return:
        """
        posItems = []
        for user in users:
            posItems.append(self.user_item_net[user].nonzero()[1])
        return posItems
