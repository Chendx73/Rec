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
    all_files = os.listdir(fpath + "/datasets/netflix_prize/mini_training_set")
    for path in all_files:
        with open(fpath + "/datasets/netflix_prize/mini_training_set/" + path,'r') as f:
            print(f.readline())
            exit(0)

transform2triple(f_path)

