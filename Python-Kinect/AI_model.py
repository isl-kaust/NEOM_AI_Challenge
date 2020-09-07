import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
import argparse
import yaml
import torch.optim as optim
import pickle
import math
import time
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
import pandas as pd
from scipy.signal import butter, filtfilt, freqz


## Model

def get_edge():
    num_node = 25
    self_link = [(i, i) for i in range(num_node)]
    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                      (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                      (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                      (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                      (22, 23), (23, 8), (24, 25), (25, 12)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    edge = self_link + neighbor_link
    center = 21 - 1
    return (edge, center)


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def get_adjacency(hop_dis, center, num_node, max_hop, dilation):
    valid_hop = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_node, num_node))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = adjacency
    A = []
    for hop in valid_hop:
        a_root = np.zeros((num_node, num_node))
        a_close = np.zeros((num_node, num_node))
        a_further = np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if hop_dis[j, i] == hop:
                    if hop_dis[j, center] == hop_dis[i, center]:
                        a_root[j, i] = normalize_adjacency[j, i]
                    elif hop_dis[j,
                                 center] > hop_dis[i,
                                                   center]:
                        a_close[j, i] = normalize_adjacency[j, i]
                    else:
                        a_further[j, i] = normalize_adjacency[j, i]
        if hop == 0:
            A.append(a_root)
        else:
            A.append(a_root + a_close)
            A.append(a_further)
    A = np.stack(A)
    return (A)


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class Model(nn.Module):
    def __init__(self, in_channels, num_class, A,
                 edge_importance_weighting, dropout):
        super().__init__()
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, dropout=dropout, residual=False),
            st_gcn(64, 64, kernel_size, 1, dropout=dropout),
            st_gcn(64, 64, kernel_size, 1, dropout=dropout),
            st_gcn(64, 64, kernel_size, 1, dropout=dropout),
            st_gcn(64, 128, kernel_size, 2, dropout=dropout),
            st_gcn(128, 128, kernel_size, 1, dropout=dropout),
            st_gcn(128, 128, kernel_size, 1, dropout=dropout),
            st_gcn(128, 256, kernel_size, 2, dropout=dropout),
            st_gcn(256, 256, kernel_size, 1, dropout=dropout),
            st_gcn(256, 256, kernel_size, 1, dropout=dropout),
        ))
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x

    def extract_feature(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        return output, feature


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def accuracy(output, labels):
    output = F.log_softmax(output, dim=1)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) * 100


## Motion Prediction


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5, axis=-1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=2)
    return y


class Predictor():
    def __init__(self, n):
        self.n = n
    #         self.m = 23-n

    def transform(self, batch):
        speed = self.computeSpeed(batch)
        pred_batch = self.predictPose(batch, speed)
        return pred_batch
    #         return torch.FloatTensor(pred_batch)

    def computeSpeed(self, batch):
        # [N,3,20,25,1]
        m = batch.shape[2] - self.n
        speed = np.zeros(batch.shape)
        speed[:, :, 1:, :, :] = batch[:, :, 1:, :, :] - batch[:, :, :-1, :, :]
        speed = butter_lowpass_filter(speed[:, :, :m, :, :], 0.5, 30, order=5, axis=2)
        return speed

    def predictPose(self, batch, speed):
        m = batch.shape[2] - self.n
        pred = np.copy(batch)
        t = 1
        for i in range(batch.shape[2] - self.n, batch.shape[2]):
            pred[:, :, i, :, :] = pred[:, :, m - 1, :, :] + t * speed[:, :, -1, :, :]
            t += 1
        return pred


## Load Model

def pre_normalization(data):
    N, C, T, V, M = data.shape
    s = np.transpose(data.copy(), [0, 4, 2, 3, 1])

    #     print('sub the center joint')
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            #             print(main_body_center.shape)
            #             print(s[i_s, i_p].shape)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask
    #             print(mask)
    #     print('Unit torso length')
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        body_unit = skeleton[0][:, 2:3, :].copy()
        #         print(body_unit.shape)
        dist_factor = 1 / np.sqrt(np.sum((body_unit ** 2), axis=-1))
        dist_factor = np.expand_dims(dist_factor, -1)
        dist_factor = np.tile(dist_factor, (1, 25, 3))
        #         print(dist_factor.shape)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            s[i_s, i_p] = np.multiply(s[i_s, i_p], dist_factor)

    s = np.transpose(s, [0, 4, 2, 3, 1])
    return s

## datapoint example



##

