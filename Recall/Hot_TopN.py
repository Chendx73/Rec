import os
import numpy as np

cwd = os.getcwd()
f_path = os.path.abspath(os.path.join(cwd, ".."))


def get_top_n(N: int = 30) -> list:
    data_fpath = "datasets/netflix_prize/output/data.txt"
    m = dict()
    fp_data = open(data_fpath, 'r')
    for i in fp_data:
        id, score = int(i.strip('\n').split(',')[1]), int(i.strip('\n').split(',')[2])
        if id not in m:
            m[id] = score
        else:
            m[id] += score
    sorted_list = sorted(m.items(), key=lambda x: x[1], reverse=True)
    topN = [x[0] for x in sorted_list[:N]]
    return topN


get_top_n()
