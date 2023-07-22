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
    def __init__(self, log):
        super(DPESol, self).__init__()

        # ESM-2
        self.esm_model, self.esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.esm_model.eval()
        log(f'Info: esm2_t33_650M_UR50D load finish', True)

        # 冻结参数
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # MPL 1 将 [batch_size, num_tokens, 1280] 拟合成 [batch_size, num_tokens, 10]
        input_size = 1280 * (protein_seq_max_len + 2)  # ESM-2模型的输出大小，并展平

        # 150528 -> 655136 -> 16384 -> 4069
        hidden_sizes = [1024 * 64, 1024 * 16, 1024 * 4, 1024]  # 隐藏层大小列表

        output_size = 1  # MLP的输出大小
        dropout_prob = 0.2  # 被丢弃的概率
        self.mpl = MLP(input_size, hidden_sizes, output_size, dropout_prob)

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

        # 改变维度大小 torch.Size([batch_size, num_tokens, 10]) -> torch.Size([batch_size, num_tokens * 10])
        protein_sequence_embedding = protein_sequence_embedding.contiguous().view(protein_sequence_embedding.size(0), -1)
        # print(protein_sequence_embedding.shape)

        # 序列长度填充到 1176 * 1280
        protein_sequence_embedding = F.pad(protein_sequence_embedding, (0, (protein_seq_max_len + 2) * 1280 - protein_sequence_embedding.size(1), 0, 0), value=0)
        # print(protein_sequence_embedding.shape)

        # 在特征上应用MLP进行进一步预测 torch.Size([batch_size, num_tokens * 10]) -> torch.Size([batch_size, 1])
        output = self.mpl(protein_sequence_embedding)

        # torch.Size([3, 9, 1]) 最后返回的结果是每个 token 一个预测值, 这不是我们想要的，想要 torch.Size([3, 1]) 所以需要改变维度大小
        # print(output.shape)

        return output


if __name__ == '__main__':
    model = DPESol()
    alphabet = model.esm_alphabet
    batch_converter = alphabet.get_batch_converter()
    # print(model)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 构建 dataset
    seq_dataset = SequenceDataset(dataset_file)

    # 划分数据集
    dataset_scale = 0.8
    train_size = int(dataset_scale * len(seq_dataset))
    train_dataset, test_dataset = random_split(seq_dataset, [train_size, len(seq_dataset) - train_size])

    # 加载数据集
    batch_size = 16
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset_collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset_collate_fn)

    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        loss_total, loss_count = 0, 0
        for step, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 序列转换
            batch_labels, batch_strs, batch_tokens = batch_converter(inputs)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            outputs = model(batch_tokens)

            # tensor[2, 1] -> tensor[2]
            outputs = torch.squeeze(outputs)
            # print(outputs.shape, targets.shape)
            # raise

            # output: [batch_size, max_tokens_len, 1], target: [batch_size, 1]
            loss = criterion(outputs, targets)

            # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
            outputs_np, targets_np = outputs.detach().numpy(), targets.numpy()

            # 计算均方根误差（RMSE）
            rmse = np.sqrt(mean_squared_error(outputs_np, targets_np))

            # 计算决定系数（R^2） 参数顺序是真实标签在前，预测结果在后
            r2 = r2_score(targets_np, outputs_np)

            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            loss_count += 1

            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}] Loss: {loss.item():.4f}, RMSE: {rmse}, R^2: {r2}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {(loss_total / loss_count):.4f}")
