import time
import typing
from typing import Tuple
import json
import os

import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
from Utils.utils_Pytorch import time_since, count_parameters
from modules_Seanny123 import Encoder, Decoder
from config import device, DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tensorGPU


def preprocess_data(data, col_names) -> Tuple[TrainData, StandardScaler]:
    """
    :param data: 数据
    :param col_names: 列名
    :return:
    """
    scaler = StandardScaler().fit(data)  # 计算均值和标准差以便后面缩放
    proc_data = scaler.transform(data)  # 缩放数据

    mask = np.ones(proc_data.shape[1], dtype=bool)
    data_cols = list(data.columns)
    for col_name in col_names:
        mask[data_cols.index(col_name)] = False

    feats = proc_data[:, mask]  # 对应 True 二维数组
    targs = proc_data[:, ~mask]  # 对应 False 二维数组

    return TrainData(feats, targs), scaler


def da_rnn(train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128):
    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size:{train_cfg.train_size:d}.")

    # 创建设置encoder参数设置
    enc_kwargs = {"input_size": train_data.feats.shape[1],
                  "hidden_size": encoder_hidden_size,
                  "T": T}
    encoder = Encoder(**enc_kwargs).to(device)
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)  # Python数据结构转换为JSON,indent表示缩进
        fi.close()

    # 创建设置decoder参数设置
    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size,
                  "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs).to(device)
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)
        fi.close()

    # encoder、decoder优化器设置
    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)

    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)
    return train_cfg, da_rnn_net


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(
        f"[{time_since(start)}] Iterations per epoch:{t_cfg.train_size * 1. / t_cfg.batch_size:3.3f}~{iter_per_epoch:d}")

    n_iter = 0  # 总训练次数
    for e_i in range(n_epochs):
        # perm_idx 长度为T的时间段的起始下标
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)  # permutation打乱数据 范围:0~train_size-T-1

        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):  # 0~train_size 步长为 batch_size
            # 返回每一个 batch_size 的 起始下标 构成的数组
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            feats, y_history, y_target = prepare_train_data(batch_idx, t_cfg, train_data)

            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
            # 第e_i个循环（每个循环iter_per_epoch个）的第 t_i // t_cfg.batch_size个
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            n_iter += 1
            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        """
        一轮训练完打印一次结果
        """
        if e_i % 1 == 0:
            y_test_pred = predict(net, train_data, t_cfg.train_size, t_cfg.batch_size, t_cfg.T, on_train=False)
            # TODO: make this mse and make it work for multiple inputs
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
            logger.info(
                f"[{time_since(start)}] Epoch{e_i:d} - train loss:{epoch_losses[e_i]}  val loss:{np.mean(np.abs(val_loss))}.")
            y_train_pred = predict(net, train_data, t_cfg.train_size, t_cfg.batch_size, t_cfg.T, on_train=True)

            plt.figure()
            plt.title(f"pred_{e_i}")
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs, label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred, label="Predicted -Train")
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                     label='Predicted - Test')
            plt.legend(loc="upper left")
            utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)

    return iter_losses, epoch_losses


def prepare_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    """
    :param batch_idx: batch_size的 起始下标 构成的数组
    :param t_cfg:
        T: 时间步
        train_size: 训练集大小
        batch_size: 一个batch大小
        loss_func:  损失函数
    :param train_data:
        feats: 特征参数数组
        targs: 预测目标数组
    :return:
    """

    # batch_size* timestep*input_size
    """
    这个地方有问题feats输入的应该是T个
    """
    # feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    feats = np.zeros((len(batch_idx), t_cfg.T, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]  # 预测的目标是T个时间步之后的值，输入的是T-1时间步的feats和历史值

    for b_i, b_idx in enumerate(batch_idx):  # i下标，idx值
        # b_slc = slice(b_idx, b_idx + t_cfg.T - 1)  # 这里切片是一个时间片(T-1个时间步)切一次 # 包左不包右
        # feats[b_i, :, :] = train_data.feats[b_slc, :]
        # y_history[b_i, :] = train_data.targs[b_slc]
        feats_slc = slice(b_idx, b_idx + t_cfg.T)
        history_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = train_data.feats[feats_slc, :]
        y_history[b_i, :] = train_data.targs[history_slc]
    return feats, y_history, y_target


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, input_data, y_history, y_target):
    t_net.enc_opt.zero_grad()  # 初始化梯度 优化器优化前-->梯度清零
    t_net.dec_opt.zero_grad()
    input_weighted, input_encoded = t_net.encoder(numpy_to_tensorGPU(input_data))
    y_pred = t_net.decoder(input_encoded, numpy_to_tensorGPU(y_history))
    y_true = numpy_to_tensorGPU(y_target)

    loss = loss_func(y_pred, y_true)
    loss.backward()  # 反向传播

    t_net.enc_opt.step()  # 计算损失更新权重
    t_net.dec_opt.step()  # 计算损失更新权重

    return loss.item()


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # the lr start from 0.001 and is reduced by 10% after each 10000 iterations 1w次迭代后衰减
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params["lr"] = enc_params["lr"] * 0.9
            dec_params["lr"] = dec_params["lr"] * 0.9


def predict(t_net: DaRnnNet, t_data: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
    out_size = t_data.targs.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))  # 训练模式的数据是train_size - T + 1个
    else:
        y_pred = np.zeros((t_data.feats.shape[0] - train_size, out_size))  # 非训练状态则 剩下的数据构成预测个数

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]  # 这里切片是每一个batch切一次
        b_len = len(batch_idx)
        # input_data = np.zeros((b_len, T - 1, t_data.feats.shape[1]))
        input_data = np.zeros((b_len, T, t_data.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_data.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                # idx = range(b_idx, b_idx + T - 1)  # 训练模型：输入当前+之后T-2个 # 包左不包右
                input_idx = range(b_idx, b_idx + T)
                history_idx = range(b_idx, b_idx + T - 1)
            else:
                # idx = range(b_idx + train_size - T, b_idx + train_size - 1)  # 非训练模式：输入之前的T-1个
                input_idx = range(b_idx + train_size - T, b_idx + train_size)
                history_idx = range(b_idx + train_size - T, b_idx + train_size - 1)


            input_data[b_i, :, :] = t_data.feats[input_idx, :]
            y_history[b_i, :] = t_data.targs[history_idx]

        y_history = numpy_to_tensorGPU(y_history)
        input_data = numpy_to_tensorGPU(input_data)
        _, input_encoded = t_net.encoder(input_data)
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()
    return y_pred


if __name__ == '__main__':
    """ 设置日志"""
    start = time.time()
    logger = utils.setup_log(tag="Debug Model")
    logger.info(f"[{time_since(start)}] Using computation device:{device}")

    """数据集读取"""
    debug = False
    dataroot = r"G:\# Project\数据集\UsingDataSet\Other_CSV\nasdaq\nasdaq100_padding.csv"
    raw_data = pd.read_csv(dataroot, nrows=100 if debug else None)
    logger.info(
        f"[{time_since(start)}] Shape of data:{raw_data.shape}. Missing in data:{raw_data.isnull().sum().sum()}.")
    targ_cols = ("NDX",)
    data, scaler = preprocess_data(raw_data, targ_cols)

    """Hyper-parameters settings"""
    save_plots = True
    n_epochs = 10
    da_rnn_kwargs = {
        "n_targs": len(targ_cols),
        "encoder_hidden_size": 64,
        "decoder_hidden_size": 64,
        "T": 10,  # ntimestep
        "learning_rate": 0.001,
        "batch_size": 128,
    }
    config, model = da_rnn(train_data=data, **da_rnn_kwargs)
    logger.info(
        f"模型参数量统计 encoder:{count_parameters(model.encoder)},decoder:{count_parameters(model.decoder)},共计:{count_parameters(model.encoder) + count_parameters(model.decoder)}")

    iter_loss, epoch_loss = train(net=model,
                                  train_data=data,
                                  t_cfg=config,
                                  n_epochs=n_epochs,
                                  save_plots=save_plots)

    final_y_pred = predict(t_net=model,
                           t_data=data,
                           train_size=config.train_size,
                           batch_size=config.batch_size,
                           T=config.T)

    plt.figure()
    plt.title("iter_loss")
    plt.semilogy(range(len(iter_loss)), iter_loss)
    # plt.plot(range(len(iter_loss)), iter_loss)
    utils.save_or_show_plot("iter_loss.png", save_plots)

    plt.figure()
    plt.title("epoch_loss")
    plt.semilogy(range(len(epoch_loss)), epoch_loss)
    # plt.plot(range(len(epoch_loss)), epoch_loss)
    utils.save_or_show_plot("epoch_loss.png", save_plots)

    plt.figure()
    plt.title("final_predicted")
    plt.plot(final_y_pred, label="Predicted")
    plt.plot(data.targs[config.train_size:], label="True")
    plt.legend(loc='upper left')
    utils.save_or_show_plot("final_predicted.png", save_plots)

    """
    保存参数与模型
    """
    with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
        # 本地回调写法 json.load()
        json.dump(da_rnn_kwargs, fi, indent=4)

    joblib.dump(scaler, os.path.join("data", "StandardScaler.pkl"))

    # 网络参数保存    nvidia-smi
    torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
    torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))
