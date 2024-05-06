import os
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


def transform2triple(fpath: str):
    """
    将训练集中所有数据->用户id,电影id,评分
    :param fpath:
    :return:
    """
    all_files = os.listdir(fpath + "/datasets/netflix_prize/mini_training_set")
    fw = open(fpath + '/datasets/netflix_prize/output/data.txt', 'w')
    flag = True
    movie_id = None
    for path in all_files:
        with open(fpath + "/datasets/netflix_prize/mini_training_set/" + path, 'r') as f:
            for i in f:
                if flag:
                    movie_id = i.split(':\n')[0]
                    flag = False
                else:
                    user_id, score = i.strip(',')[0], i.strip(',')[1]
                    fw.write(user_id + "," + movie_id + "," + score + "\n")
    fw.close()


transform2triple(f_path)
