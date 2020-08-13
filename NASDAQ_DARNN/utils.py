import logging
import os

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from config import device

def setup_log(tag="VOC_TOPICS"):
    # 设置log记录
    # create logger
    logger=logging.getLogger(tag)   # 获得名字

    logger.propagate=False
    logger.setLevel(logging.DEBUG) # 设置日志等级

    # create console handler and set level to debug
    ch=logging.StreamHandler() # 在控制台输出，这里没设置写入文件
    ch.setLevel(logging.DEBUG)

    # create formatter
    # formatter=logging.Formatter('%(asctime)s - %(name)s:%(levelname)s: %(message)s',"%Y-%m-%d-%H:%M:%S")
    formatter=logging.Formatter('%(name)s:%(levelname)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger


def save_or_show_plot(file_nm:str,save:bool):
    # 画图或保存
    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), "data", file_nm))
    else:
        plt.show()


def numpy_to_tensorGPU(x):
    # 张量转移到gpu上
    # return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))
    return torch.from_numpy(x).type(torch.FloatTensor).to(device)