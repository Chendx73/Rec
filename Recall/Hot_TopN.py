import os
import numpy as np

N = 30
cwd = os.getcwd()
f_path = os.path.abspath(os.path.join(cwd, ".."))
train_fpath = f_path + "/datasets/netflix_prize/train.txt"
m = dict()
