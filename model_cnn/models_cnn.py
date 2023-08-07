# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from args import seq_max_len, acid_numb


# 模型参数
class CnnConfig:
    def __init__(self, inc, outc, ks, pks):
        # 输入数据高度，RGB 3
        self.in_channel = inc

        # 卷积核个数 16
        self.out_channel = outc

        # 卷积核大小 (3, 25)
        self.kernel_size = ks

        # 池化 (2, 2)
        self.pool_kernel_size = pks


class ModelConfig:
    def __init__(self):
        # 原始数据：batch*seq_len*1,
        self.seq_len = seq_max_len

        # 嵌入参数: 输入 batch*seq_len*1 -> batch*seq_len*64
        self.num_embeddings = acid_numb + 1
        self.embedding_dim = 64

        # 经过变换：-> batch*64*seq_len，输入通道都是

        # 卷积参数
        self.cnn = []

        in_channel = 64
        out_channel = 128
        kernel_size = [i for i in range(2, 16)]
        # seq_len - kernel_size + 1
        pool_kernel_size = self.seq_len - np.array(kernel_size) + 1

        for i in range(len(kernel_size)):
            self.cnn.append(CnnConfig(in_channel, out_channel,
                                      kernel_size[i], pool_kernel_size[i]))

        # 全连接
        self.dropout_rate = 0.2
        self.in_features = out_channel * len(kernel_size)
        self.num_class = 2


class ProteinEmbeddingCNN(nn.Module):
    def __init__(self):
        super(ProteinEmbeddingCNN, self).__init__()
        # args
        ModuleList_kernel_list = range(2, 10)
        Sequential_channel_list = [1280, 1920, 1280]

        models = []
        for kernel_size in ModuleList_kernel_list:
            layers = []
            for channel_index in range(len(Sequential_channel_list)):
                layers.append(nn.Conv1d(in_channels=Sequential_channel_list[channel_index],
                                        out_channels=Sequential_channel_list[channel_index + 1],
                                        kernel_size=kernel_size,
                                        padding=int(kernel_size / 2)))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool1d(kernel_size=kernel_size))

            models.append(nn.Sequential(*layers))

        self.convs = nn.ModuleList(models)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = [torch.max(conv(x), dim=2)[0] for conv in self.convs]

# 构建模型
class KMersCNN(nn.Module):
    # https://blog.csdn.net/sunny_xsc1994/article/details/82969867
    def __init__(self):
        super(KMersCNN, self).__init__()

        self.args = ModelConfig()

        # https://www.jianshu.com/p/63e7acc5e890
        # 词嵌入: 词典大小为 25, 嵌入维度 64
        self.embedding = nn.Embedding(num_embeddings=self.args.num_embeddings,
                                      embedding_dim=self.args.embedding_dim)

        # 卷积
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=cnn.in_channel,
                          out_channels=cnn.out_channel,
                          kernel_size=cnn.kernel_size), nn.ReLU(),
                nn.MaxPool1d(kernel_size=cnn.pool_kernel_size))
            for cnn in self.args.cnn
        ])

        # 全连接
        self.fc = nn.Linear(in_features=self.args.in_features,
                            out_features=self.args.num_class)

    def forward(self, x):
        x = x.long()

        embed = self.embedding(x)

        x = embed.permute(0, 2, 1)

        out = [conv(x) for conv in self.convs]

        out = torch.cat(out, dim=1)

        out = out.view(-1, out.size(1))

        out = F.dropout(input=out, p=self.args.dropout_rate)
        out = self.fc(out)

        return out


# Test K_MERS_CNN class
if __name__ == '__main__':
    model = KMersCNN()
    print(model)
