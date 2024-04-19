from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.SelfLog import Log

log = Log()


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
        log.logger.info("init dataset")


class Loader(BasicDataset):
    def __init__(self, path):
        super().__init__()
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
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
        print('dataset Sparsity: {}'.format())
        log.logger.info('{} interactions for training'.format(self.train_data_size))
        log.logger.info('{} interactions for testing'.format(self.test_data_size))
