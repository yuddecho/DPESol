import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from dataset import SequenceDataset, dataset_collate_fn
from DPESol.args import dataset_file, protein_seq_max_len


# 1. 多层感知机
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob):
        """
        :param input_size: input_size = 1280  # ESM-2模型的输出大小
        :param hidden_sizes: hidden_sizes = [256, 128]  # 隐藏层大小列表
        :param output_size: output_size = 1  # MLP的输出大小
        """
        super(MLP, self).__init__()
        layers = []

        # 最后一层不添加
        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # sizes = [input_size] + hidden_sizes + [output_size]
        # for i in range(len(sizes)-1):
        #     layers.append(nn.Linear(sizes[i], sizes[i+1]))
        #     layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 整体预测网络
class DPESol(nn.Module):
    def __init__(self, esm_model):
        super(DPESol, self).__init__()

        # ESM-2
        self.esm_model = esm_model

        # 冻结参数
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # MPL 1 将 [batch_size, num_tokens, 1280] 拟合成 [batch_size, num_tokens, 10]
        input_size = 1280  # ESM-2模型的输出大小，并展平
        hidden_sizes = [512, 128]  # 隐藏层大小列表
        output_size = 10  # MLP的输出大小
        dropout_prob = 0.2  # 被丢弃的概率
        self.mpl_1 = MLP(input_size, hidden_sizes, output_size, dropout_prob)

        # MPL 2
        input_size = 10 * (protein_seq_max_len + 2)  # ESM-2模型的输出大小，并展平

        # 150528 -> 655136 -> 16384 -> 4069
        # hidden_sizes = [1024 * 64, 1024 * 16, 1024 * 4, 1024]  # 隐藏层大小列表

        # 11760 1024 128
        hidden_sizes = [1024, 128]  # 隐藏层大小列表

        output_size = 1  # MLP的输出大小
        dropout_prob = 0.2  # 被丢弃的概率
        self.mpl_2 = MLP(input_size, hidden_sizes, output_size, dropout_prob)

        # 拼接
        # self.esm_model.add_module("mlp", self.mpl)

    def get_esm_alphabet(self):
        return self.esm_alphabet

    def forward(self, x):
        # 使用ESM-2编码蛋白质序列
        with torch.no_grad():
            res = self.esm_model(x, repr_layers=[33])

        # 提取预训练模型的最后一层特征作为序列表示 torch.Size([batch_size, max_token_len, 1280])
        # 序列会被统一到最大长度，max_token_len = max_seq_len + 2 (序列开始和结束添加了标记)，每个 token 都被编码成了 1280 的 tensor
        protein_sequence_embedding = res["representations"][33]
        # print(protein_sequence_embedding.shape)

        # 先将 1280 拟合到 10 torch.Size([batch_size, num_tokens, 1280]) -> torch.Size([batch_size, num_tokens, 10])
        protein_sequence_embedding = self.mpl_1(protein_sequence_embedding)
        # print(protein_sequence_embedding.shape)

        # 改变维度大小 torch.Size([batch_size, num_tokens, 10]) -> torch.Size([batch_size, num_tokens * 10])
        protein_sequence_embedding = protein_sequence_embedding.contiguous().view(protein_sequence_embedding.size(0), -1)
        # print(protein_sequence_embedding.shape)

        # 序列长度填充到 1176 * 10
        protein_sequence_embedding = F.pad(protein_sequence_embedding, (0, (protein_seq_max_len + 2) * 10 - protein_sequence_embedding.size(1), 0, 0), value=0)
        # print(protein_sequence_embedding.shape)

        # 在特征上应用MLP进行进一步预测 torch.Size([batch_size, num_tokens * 10]) -> torch.Size([batch_size, 1])
        output = self.mpl_2(protein_sequence_embedding)

        # torch.Size([3, 9, 1]) 最后返回的结果是每个 token 一个预测值, 这不是我们想要的，想要 torch.Size([3, 1]) 所以需要改变维度大小
        # print(output.shape)

        return output
