import numpy as np
import torch


def intertemporal_recurrence_matrix(data):
    N = len(data)
    data = np.array(data)
    inter_recurrence_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if data[i] == 0:
                data[i] += 1e-5
            if data[j] == 0:
                data[j] += 1e-5
            inter_recurrence_matrix[i, j] = np.log(data[i] / data[j])
    return inter_recurrence_matrix


def distance_matrix(data, dimension, delay, norm):
    N = int(len(data) - (dimension-1) * delay)
    distance_matrix = np.zeros((N, N), dtype="float32")
    if norm == 'manhattan':
        for i in range(N):
            for j in range(i, N, 1):
                temp = 0.0
                for k in range(dimension):
                    temp += np.abs(data[i+k*delay] - data[j+k*delay])
                distance_matrix[i, j] = distance_matrix[j, i] = temp
    elif norm == 'euclidean':
        for i in range(N):
            for j in range(i, N, 1):
                temp = 0.0
                for k in range(dimension):
                    temp += np.power(data[i+k*delay] - data[j+k*delay], 2)
                distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(temp)
    elif norm == 'supremum':
        temp = np.zeros(dimension)
        for i in range(N):
            for j in range(i, N, 1):
                for k in range(dimension):
                    temp[k] = np.abs(data[i+k*delay] - data[j+k*delay])
                distance_matrix[i, j] = distance_matrix[j, i] = np.max(temp)
    return distance_matrix


def recurrence_matrix(data, dimension, delay, threshold, norm):
    recurrence_matrix = distance_matrix(data, dimension, delay, norm)
    N = len(recurrence_matrix[:, 0])
    for i in range(N):
        for j in range(i, N, 1):
            if recurrence_matrix[i, j] <= threshold:
                recurrence_matrix[i, j] = recurrence_matrix[j, i] = 1
            else:
                recurrence_matrix[i, j] = recurrence_matrix[j, i] = 0
    return recurrence_matrix.astype(int)


def de_irp(data, init_value):
    """
    reconstruct the origin data from IRP value with a init value
    :param data:
    :param init_value:
    :return:
    """
    seq_length = data.shape[0]
    res = np.zeros(seq_length)
    res[0] = init_value
    for i in range(1, seq_length):
        res[i] = init_value / (np.exp(data[i]))

    return res


def de_norm(a, c, x):
    """
    逆标准化，将数据从[-1,1]还原回原有值域
    :param a: 标准化系数a
    :param c: 标准化系数c
    :param x: 数据
    :return:
    """
    res = c * x + a

    return res


def de_tanh(x):
    y = 0.5 * torch.log((1 + x) / (1 - x) + 1e-7)

    return y


def de_irpv2(data, init_value):
    """
    使用矩阵的每一行进行逆rp化，后取均值
    :param data: rp值矩阵
    :param init_value: 初始值
    :return:
    """
    seq_length = data.shape[0]
    res_matrix = np.zeros((seq_length, seq_length))
    for i in range(seq_length):
        res_matrix[i][i] = init_value * np.exp(data[i][0])
        for j in range(seq_length):
            res_matrix[i][j] = res_matrix[i][i] / np.exp(data[i][j])
    return res_matrix.mean(axis=0)
