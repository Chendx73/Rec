import os
import random
import json
import numpy as np

cwd = os.getcwd()
f_path = os.path.abspath(os.path.join(cwd, ".."))


def process_movie_title(fpath: str) -> dict:
    """
    电影的元数据->电影id:(年份,电影名)
    :param fpath:
    :return:
    """
    data = fpath + "/datasets/netflix_prize/movie_titles.txt"
    metadata_map = dict()
    with open(data, 'r', encoding='utf8', errors='ignore') as f:
        for i in f:
            tmp = i.strip('\n').split(',')
            metadata_map[int(tmp[0])] = (tmp[1], tmp[2])
    return metadata_map


def concate_id_name(fpath: str):
    movie_fp = fpath + '/datasets/netflix_prize/movie_titles.txt'
    data_fp = fpath + '/datasets/netflix_prize/output/data.txt'
    movie_dic = {}
    all_dic_list = []
    fr_movie = open(movie_fp, 'r', encoding='utf8', errors='ignore')
    fr_data = open(data_fp, 'r')
    for i in fr_movie:
        id, title = i.strip('\n').split(',')[0], i.strip('\n').split(',')[2]
        movie_dic[id] = title
    for i in fr_data:
        tmp = {}
        id, score = i.strip('\n').split(',')[1], i.strip('\n').split(',')[2]
        tmp['id'], tmp['score'], tmp['title'] = id, score, movie_dic[id]
        json.dump()
        all_dic_list.append(tmp)
    json.dump(all_dic_list, open(fpath + '/datasets/netflix_prize/output/all_data.jsonl', 'w'), ensure_ascii=False)


def transform2triple(fpath: str) -> None:
    """
    将数据集中所有数据->(用户id,电影id,评分)
    :param fpath:
    :return:
    """
    all_files = os.listdir(fpath + "/datasets/netflix_prize/mini_training_set")
    fw = open(fpath + '/datasets/netflix_prize/output/data.txt', 'w')
    movie_id = None
    for path in all_files:
        flag = True
        with open(fpath + "/datasets/netflix_prize/mini_training_set/" + path, 'r') as f:
            for i in f:
                if flag:
                    movie_id = i.split(':\n')[0]
                    flag = False
                else:
                    if i.strip(',')[1].isalnum():  # 原始数据存在脏数据,目前采用丢弃的策略
                        user_id, score = i.strip(',')[0], i.strip(',')[1]
                        fw.write(user_id + "," + movie_id + "," + score + "\n")
                    else:
                        continue
    fw.close()


def sample_dataset(fpath: str):
    """
    将数据集中(用户id,电影id,评分)按7:3比例随机分为训练集和测试集
    :param fpath:
    :return:
    """
    data_fp = fpath + '/datasets/netflix_prize/output/data.txt'
    train_fp = fpath + '/datasets/netflix_prize/output/train.txt'
    test_fp = fpath + '/datasets/netflix_prize/output/test.txt'
    f_train = open(train_fp, 'w')
    f_test = open(test_fp, 'w')
    f_data = open(data_fp, 'r')
    for i in f_data:
        rand = random.random()
        if rand > 0.7:
            f_test.write(i)
        else:
            f_train.write(i)
    f_train.close()
    f_test.close()


concate_id_name(f_path)
