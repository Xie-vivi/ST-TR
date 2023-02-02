# script to read, prepare and display data

import numpy as np
from scipy import misc
import os
import matplotlib.pyplot as plt
import csv


# Function to load a csv skeleton file sequence (train)
# It returns idx, sequence label and sizesequence (with non zero padding frames)
def read_csv_train_infos(file):
    idx = []
    labels = []
    sizesequences = []

    with open(file, 'r') as f:
        data = csv.reader(f, delimiter=' ')
        for d in data:
            j = d[0].split(',')
            idx.append(int(j[0]))
            labels.append(int(j[1]))
            sizesequences.append(int(j[2]))
    return idx, labels, sizesequences


# Function to load a csv skeleton file sequence (test)
# It returns idx and sizesequence (with non zero padding frames)
def read_csv_test_infos(file):
    idx = []
    sizesequences = []

    with open(file, 'r') as f:
        data = csv.reader(f, delimiter=' ')
        for d in data:
            j = d[0].split(',')
            idx.append(int(j[0]))
            sizesequences.append(int(j[1]))
    return idx, sizesequences


# Function to display a sequence of 2d skeleton
def diplay_skeleton(skeletons_image, size):
    ####

    # Idx of the bones in the hand skeleton to display it.

    bones = np.array([
        [0, 1],
        [0, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [1, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [1, 14],
        [14, 15],
        [15, 16],
        [16, 17],
        [1, 18],
        [18, 19],
        [19, 20],
        [20, 21]
    ]
    )
    skeletons_image = np.reshape(skeletons_image,
                                 (skeletons_image.shape[0], skeletons_image.shape[1] * skeletons_image.shape[2]))
    pngDepthFiles = np.zeros([size, 480, 640])
    skeletons_display = np.zeros([size, 2, 2, 21])

    for id_image in range(0, size):
        # pngDepthFiles[id_image,:] = misc.imread(path_gesture+str(id_image)+'_depth.png')

        x = np.zeros([2, bones.shape[0]])
        y = np.zeros([2, bones.shape[0]])

        ske = skeletons_image[id_image, :]

        for idx_bones in range(0, bones.shape[0]):
            joint1 = bones[idx_bones, 0]
            joint2 = bones[idx_bones, 1]

            pt1 = ske[joint1 * 2:joint1 * 2 + 2]
            pt2 = ske[joint2 * 2:joint2 * 2 + 2]

            x[0, idx_bones] = pt1[0]
            x[1, idx_bones] = pt2[0]
            y[0, idx_bones] = pt1[1]
            y[1, idx_bones] = pt2[1]

        skeletons_display[id_image, 0, :, :] = x
        skeletons_display[id_image, 1, :, :] = y

    for id_image in range(0, size):
        plt.clf()
        plt.imshow(pngDepthFiles[id_image, :])
        plt.plot(skeletons_display[id_image, 0, :, :], skeletons_display[id_image, 1, :, :], linewidth=2.5)
        plt.pause(0.01)

import torch

import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import os

sys.path.append("/kaggle/working/ST-TR/code")

import numpy as np
from st_gcn.graph import tools


# Edge format: (origin, neighbor)
num_node = 22
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1), (0, 2), (2, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 8),
                                  (8, 9), (1, 10), (10, 11), (11, 12), (12, 13), (1, 14), (14, 15),
                                  (15, 16), (16, 17), (1, 18), (18, 19), (19, 20), (20, 21)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix
    For more information, please refer to the section 'Partition Strategies' in our paper.
    """

    def __init__(self, labeling_mode='uniform'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        return A


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import tqdm

sys.path.append("/kaggle/working/ST-TR/code/st_gcn/net")
from st_gcn.net.temporal_transformer_windowed import tcn_unit_attention_block
from st_gcn.net.temporal_transformer import tcn_unit_attention

from st_gcn.net.gcn_attention import gcn_unit_attention
from st_gcn.net.net import Unit2D, conv_init, import_class
from st_gcn.net.unit_gcn import unit_gcn
from st_gcn.net.unit_agcn import unit_agcn

window_size = 60


class Model(nn.Module):
    """ Spatial temporal graph convolutional networks
                        for skeleton-based action recognition.

    Input shape:
        Input shape should be (N, C, T, V, M)
        where N is the number of samples,
              C is the number of input channels,
              T is the length of the sequence,
              V is the number of joints or graph nodes,
          and M is the number of people.

    Arguments:
        About shape:
            channel (int): Number of channels in the input data
            num_class (int): Number of classes for classification
            window_size (int): Length of input sequence
            num_point (int): Number of joints or graph nodes
            num_person (int): Number of people
        About net:
            use_data_bn: If true, the data will first input to a batch normalization layer
            backbone_config: The structure of backbone networks
        About graph convolution:
            graph: The graph of skeleton, represtented by a adjacency matrix
            graph_args: The arguments of graph
            mask_learning: If true, use mask matrixes to reweight the adjacency matrixes
            use_local_bn: If true, each node in the graph have specific parameters of batch normalzation layer
        About temporal convolution:
            multiscale: If true, use multi-scale temporal convolution
            temporal_kernel_size: The kernel size of temporal convolution
            dropout: The drop out rate of the dropout layer in front of each temporal convolution layer

    """

    def __init__(self,
                 channel=3,
                 num_class=28,
                 window_size=window_size,
                 num_point=22,
                 num_person=1,
                 mask_learning=True,
                 use_data_bn=True,
                 attention=True,
                 only_attention=True,
                 tcn_attention=False,
                 data_normalization=True,
                 skip_conn=True,
                 weight_matrix=2,
                 only_temporal_attention=True,
                 bn_flag=True,
                 attention_3=False,
                 kernel_temporal=9,
                 more_channels=False,
                 double_channel=True,
                 drop_connect=True,
                 concat_original=True,
                 all_layers=False,
                 adjacency=False,
                 agcn=False,
                 dv=0.25,
                 dk=0.25,
                 Nh=8,
                 n=4,
                 dim_block1=10,
                 dim_block2=30,
                 dim_block3=75,
                 relative=False,
                 #   graph: st_gcn.graph.NTU_RGB_D
                 visualization=False,
                 #     device=1,
                 graph_args={
                     "labeling_mode": 'spatial'
                 },
                 backbone_config=None,
                 graph=None,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 device=1
                 #                  agcn = True
                 #                  multiscale=False
                 ):
        super(Model, self).__init__()
        #         if graph is None:
        #             raise ValueError()
        #         else:
        #             Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        # self.A = torch.from_numpy(self.graph.A).float().cuda(0)
        # self.A = torch.from_numpy(self.graph.A).float()
        # self.A = self.graph.A
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))

        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.multiscale = multiscale
        self.attention = attention
        self.tcn_attention = tcn_attention
        self.drop_connect = drop_connect
        self.more_channels = more_channels
        self.concat_original = concat_original
        self.all_layers = all_layers
        self.dv = dv
        self.num = n
        self.Nh = Nh
        self.dk = dk
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.visualization = visualization
        self.double_channel = double_channel
        self.adjacency = adjacency

        # Different bodies share batchNorm parameters or not
        self.M_dim_bn = True

        if self.M_dim_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_point * num_person)
        else:
            self.data_bn = nn.BatchNorm1d(channel * num_point)

        if self.all_layers:
            if not self.double_channel:
                self.starting_ch = 64
            else:
                self.starting_ch = 128
        else:
            if not self.double_channel:
                self.starting_ch = 128
            else:
                self.starting_ch = 256
        default_backbone_all_layers = [(3, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,
                                                                                           2), (128, 128, 1),
                                       (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]

        default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,
                                                                    2), (128, 128, 1),
                            (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]
        kwargs = dict(
            A=self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
            dropout=dropout,
            kernel_size=temporal_kernel_size,
            attention=attention,
            only_attention=only_attention,
            tcn_attention=tcn_attention,
            only_temporal_attention=only_temporal_attention,
            attention_3=attention_3,
            relative=relative,
            weight_matrix=weight_matrix,
            device=device,
            more_channels=self.more_channels,
            drop_connect=self.drop_connect,
            data_normalization=self.data_normalization,
            skip_conn=self.skip_conn,
            adjacency=self.adjacency,
            starting_ch=self.starting_ch,
            visualization=self.visualization,
            all_layers=self.all_layers,
            dv=self.dv,
            dk=self.dk,
            Nh=self.Nh,
            num=n,
            dim_block1=dim_block1,
            dim_block2=dim_block2,
            dim_block3=dim_block3,
            num_point=num_point,
            agcn=agcn
        )

        if self.multiscale:
            unit = TCN_GCN_unit_multiscale
        else:
            unit = TCN_GCN_unit

        # backbone
        if backbone_config is None:
            if self.all_layers:
                backbone_config = default_backbone_all_layers
            else:
                backbone_config = default_backbone
        self.backbone = nn.ModuleList([
            unit(in_c, out_c, stride=stride, **kwargs)
            for in_c, out_c, stride in backbone_config
        ])
        if self.double_channel:
            backbone_in_c = backbone_config[0][0] * 2
            backbone_out_c = backbone_config[-1][1] * 2
        else:
            backbone_in_c = backbone_config[0][0]
            backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size
        backbone = []
        for i, (in_c, out_c, stride) in enumerate(backbone_config):
            if self.double_channel:
                in_c = in_c * 2
                out_c = out_c * 2
            if i == 3 and concat_original:
                backbone.append(unit(in_c + channel, out_c, stride=stride, last=i == len(default_backbone) - 1,
                                     last_graph=(i == len(default_backbone) - 1), layer=i, **kwargs))
            else:
                backbone.append(unit(in_c, out_c, stride=stride, last=i == len(default_backbone) - 1,
                                     last_graph=(i == len(default_backbone) - 1), layer=i, **kwargs))
            if backbone_out_t % stride == 0:
                backbone_out_t = backbone_out_t // stride
            else:
                backbone_out_t = backbone_out_t // stride + 1
        self.backbone = nn.ModuleList(backbone)
        print("self.backbone: ", self.backbone)
        for i in range(0, len(backbone)):
            pytorch_total_params = sum(p.numel() for p in self.backbone[i].parameters() if p.requires_grad)
            print(pytorch_total_params)

        # head

        if not all_layers:
            if not agcn:
                self.gcn0 = unit_gcn(
                    channel,
                    backbone_in_c,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn)
            else:
                self.gcn0 = unit_agcn(
                    channel,
                    backbone_in_c,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn)

            self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9)

        # tail
        self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.gap_size = backbone_out_t
        self.fcn = nn.Conv1d(backbone_out_c, num_class, kernel_size=1)
        conv_init(self.fcn)

    def forward(self, x, label=1, name=1):
        torch.autograd.set_detect_anomaly(True)

        #         1900 171 22 3
        #         N T V C M
        print(x.shape)
        x = x[:, :, :, :, None].clone()
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        N, C, T, V, M = x.size()
        print(x.shape)
        if (self.concat_original):
            x_coord = x
            x_coord = x_coord.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)

        # data bn
        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
            x = self.data_bn(x)
            # to (N*M, C, T, V)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        else:
            # from (N, C, T, V, M) to (N*M, C, T, V)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # model
        if not self.all_layers:
            x = self.gcn0(x, label, name)
            x = self.tcn0(x)

        for i, m in enumerate(self.backbone):
            if i == 3 and self.concat_original:
                x = m(torch.cat((x, x_coord), dim=1), label, name)
            else:
                x = m(x, label, name)

        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))

        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)

        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2])

        # C fcn
        x = self.fcn(x)
        x = F.avg_pool1d(x, x.size()[2:])
        x = x.view(N, self.num_class)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 A,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 relative,
                 device,
                 attention_3,
                 dv,
                 dk,
                 Nh,
                 num,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 num_point,
                 weight_matrix,
                 more_channels,
                 drop_connect,
                 starting_ch,
                 all_layers,
                 adjacency,
                 data_normalization,
                 visualization,
                 skip_conn,
                 layer=0,
                 kernel_size=9,
                 stride=1,
                 dropout=0.5,
                 use_local_bn=False,
                 mask_learning=False,
                 last=False,
                 last_graph=False,
                 agcn=False
                 ):
        super(TCN_GCN_unit, self).__init__()
        half_out_channel = out_channel / 2
        self.A = A

        self.V = A.shape[-1]
        self.C = in_channel
        self.last = last
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.num_point = num_point
        self.adjacency = adjacency
        self.last_graph = last_graph
        self.layer = layer
        self.stride = stride
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.device = device
        self.all_layers = all_layers
        self.more_channels = more_channels

        if (out_channel >= starting_ch and attention or (self.all_layers and attention)):

            self.gcn1 = gcn_unit_attention(in_channel, out_channel, dv_factor=dv, dk_factor=dk, Nh=Nh,
                                           complete=True,
                                           relative=relative, only_attention=only_attention, layer=layer, incidence=A,
                                           bn_flag=True, last_graph=self.last_graph, more_channels=self.more_channels,
                                           drop_connect=self.drop_connect, adjacency=self.adjacency, num=num,
                                           data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                           visualization=self.visualization, num_point=self.num_point)
        else:

            if not agcn:
                self.gcn1 = unit_gcn(
                    in_channel,
                    out_channel,
                    A,
                    use_local_bn=use_local_bn,
                    mask_learning=mask_learning)
            else:
                self.gcn1 = unit_agcn(
                    in_channel,
                    out_channel,
                    A,
                    use_local_bn=use_local_bn,
                    mask_learning=mask_learning)

        if (out_channel >= starting_ch and tcn_attention or (self.all_layers and tcn_attention)):

            if out_channel <= starting_ch and self.all_layers:
                self.tcn1 = tcn_unit_attention_block(out_channel, out_channel, dv_factor=dv,
                                                     dk_factor=dk, Nh=Nh,
                                                     relative=relative, only_temporal_attention=only_temporal_attention,
                                                     dropout=dropout,
                                                     kernel_size_temporal=9, stride=stride,
                                                     weight_matrix=weight_matrix, bn_flag=True, last=self.last,
                                                     layer=layer,
                                                     device=self.device, more_channels=self.more_channels,
                                                     drop_connect=self.drop_connect, n=num,
                                                     data_normalization=self.data_normalization,
                                                     skip_conn=self.skip_conn,
                                                     visualization=self.visualization, dim_block1=dim_block1,
                                                     dim_block2=dim_block2, dim_block3=dim_block3,
                                                     num_point=self.num_point)
            else:
                self.tcn1 = tcn_unit_attention(out_channel, out_channel, dv_factor=dv,
                                               dk_factor=dk, Nh=Nh,
                                               relative=relative, only_temporal_attention=only_temporal_attention,
                                               dropout=dropout,
                                               kernel_size_temporal=9, stride=stride,
                                               weight_matrix=weight_matrix, bn_flag=True, last=self.last,
                                               layer=layer,
                                               device=self.device, more_channels=self.more_channels,
                                               drop_connect=self.drop_connect, n=num,
                                               data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                               visualization=self.visualization, num_point=self.num_point)



        else:
            self.tcn1 = Unit2D(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                dropout=dropout,
                stride=stride)
        if ((in_channel != out_channel) or (stride != 1)):
            self.down1 = Unit2D(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x, label, name):
        # N, C, T, V = x.size()
        x = self.tcn1(self.gcn1(x, label, name)) + (x if
                                                    (self.down1 is None) else self.down1(x))

        return x


class TCN_GCN_unit_multiscale(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size=9,
                 stride=1,
                 **kwargs):
        super(TCN_GCN_unit_multiscale, self).__init__()
        self.unit_1 = TCN_GCN_unit(
            in_channels,
            out_channels / 2,
            A,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs)
        self.unit_2 = TCN_GCN_unit(
            in_channels,
            out_channels - out_channels / 2,
            A,
            kernel_size=kernel_size * 2 - 1,
            stride=stride,
            **kwargs)

    def forward(self, x):
        return torch.cat((self.unit_1(x), self.unit_2(x)), dim=1)


import csv
import torch
import random


class HGRDataset(torch.utils.data.Dataset):
    def __init__(self, set_type, data, sequencesLen, window_size, labels=None):
        self.data = data
        self.set_type = set_type
        self.labels = labels
        self.sequencesLen = sequencesLen
        self.window_size = window_size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        seqLen = self.sequencesLen[index]  # 77 winsize=100
        # processing
        #             data_numpy = tools.random_choose(data_numpy, self.window_size)
        #         if seqLen < self.window_size:
        data_numpy = data_numpy[:self.window_size]
        #         else:
        #             num_time = seqLen//self.window_size
        #             if num_time == 1:
        #                 begin = random.randint(0, seqLen-self.window_size)
        #                 data_numpy = data_numpy[begin:begin+self.window_size]
        #             else:
        #                 begin = random.randint(0, seqLen-self.window_size*num_time)
        #                 indexs = [begin+i*num_time for i in range(self.window_size)]
        #                 data_numpy = data_numpy[indexs]
        if self.set_type == 'train' or self.set_type == 'valid':
            label = self.labels[index]
            return torch.from_numpy(data_numpy).float(), torch.tensor(label - 1)
        if self.set_type == 'test':
            return torch.from_numpy(data_numpy).float()

from code.main import Processor

if __name__ == '__main__':
    train_data_path = "./skeletons_world_train.npy"
    train_label_path = './infos_train.csv'
    test_label_path = './infos_test.csv'
    X_test = np.load("./skeletons_world_test.npy", allow_pickle=True)
    X_train = np.load("./skeletons_world_train.npy", allow_pickle=True)
    train_ids, train_labels, train_seqlens = read_csv_train_infos(train_label_path)
    test_ids, test_seqlens = read_csv_test_infos(test_label_path)
    # test_seqlens = np.array(test_seqlens)
    train_labels = np.array(train_labels)
    train_seqlens = np.array(train_seqlens)
    full_index = [i for i in range(len(train_ids))]
    random.shuffle(full_index)
    # print(full_index[:3])
    train_index = random.sample(full_index, int(0.8 * len(full_index)))
    # print(nptrain_labels)
    valid_index = [i for i in full_index if i not in train_index]
    # print(valid_index,train_index)
    X_train, X_valid = X_train[train_index], X_train[valid_index]
    train_labels, valid_labels = train_labels[train_index], train_labels[valid_index]
    train_seqlens, valid_seqlens = train_seqlens[train_index], train_seqlens[valid_index]
    window_size = 60
    train_dataset = HGRDataset(set_type="train", data=X_train, labels=train_labels, sequencesLen=train_seqlens,
                               window_size=window_size)
    valid_dataset = HGRDataset(set_type="valid", data=X_valid, labels=valid_labels, sequencesLen=valid_seqlens,
                               window_size=window_size)
    test_dataset = HGRDataset(set_type="test", data=X_test, sequencesLen=test_seqlens, window_size=window_size)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
    hgr_model = Model()
    processor = Processor("code/config/st_gcn/hgr/train.yaml",hgr_model,train_dataset,valid_dataset)
    processor.start()



