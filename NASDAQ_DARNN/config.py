import torch
import collections
import typing
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 具名元组 namedtuple，也就是构造一个带字段名的元组
DaRnnNet = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"])


class TrainConfig(typing.NamedTuple):
    # 创建训练参数设置类型
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable  # 检查一个对象是否可调用


class TrainData(typing.NamedTuple):
    # 训练参数类型设置
    feats: np.ndarray
    targs: np.ndarray
