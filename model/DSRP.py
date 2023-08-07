import os
import sys
import tqdm
import random
import pickle
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import mean_squared_error, r2_score

from DPESol.args import root


class SequenceDataset(Dataset):
    def __init__(self, esol_file, dna_embedding_file, protein_embedding_file):
        self.source_file = esol_file

        # read data
        self.gene_name_list = []
        self.gene_solubility_dict = {}

        # embedding
        self.dna_embedding, self.protein_embedding = {}, {}

        # init
        self._read_file(dna_embedding_file, protein_embedding_file)

    def _read_file(self, dna_embedding_file, protein_embedding_file):
        # gene name and solu
        with open(self.source_file, 'r', encoding='utf-8') as r:
            rows = r.readlines()
            for row in rows:
                row = row.strip()
                row_list = row.split(',')

                # get data
                gene_name, solubility, nucle_seq, acid_seq = row_list[0], row_list[1], row_list[2], row_list[3]

                # gene name
                self.gene_name_list.append(gene_name)
                # gene: solu
                self.gene_solubility_dict[gene_name] = float(solubility)

        # embedding
        with open(dna_embedding_file, 'rb') as r:
            self.dna_embedding = pickle.load(r)

        with open(protein_embedding_file, 'rb') as r:
            self.protein_embedding = pickle.load(r)

    def __getitem__(self, index):
        _gene_name = self.gene_name_list[index]

        solu = self.gene_solubility_dict.get(_gene_name, None)
        if solu is None:
            raise ValueError(f'Error: {_gene_name} no solu value')
        solu_tensor = torch.tensor(solu)

        dna_tensor = self.dna_embedding.get(_gene_name, None)
        if dna_tensor is None:
            raise ValueError(f'Error: {_gene_name} no dna tensor')

        protein_tensor = self.protein_embedding.get(_gene_name, None)
        if protein_tensor is None:
            raise ValueError(f'Error: {_gene_name} no protein_tensor')

        concatenated_tensor = torch.cat((dna_tensor, protein_tensor), dim=0)

        return concatenated_tensor, solu_tensor

    def __len__(self):
        return len(self.gene_name_list)


# 多层感知机
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
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # sizes = [input_size] + hidden_sizes + [output_size]
        # for i in range(len(sizes)-1):
        #     layers.append(nn.Linear(sizes[i], sizes[i+1]))
        #     layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        # 初试化参数
        for index_id in range(len(hidden_sizes) + 1):
            nn.init.kaiming_uniform_(self.mlp[index_id * 4].weight, nonlinearity='relu')

    def forward(self, x):
        return self.mlp(x)


class Args:
    def __init__(self, _parse_args):
        # 设置参数，保存模型数据，保存日志，设置随机数种子

        # 0. 设置随机数种子
        self.seed = 2023
        self._set_seed()

        # 1. 设置模型参数
        self.curr_epoch = 0
        self.num_epochs = _parse_args.num_epochs
        self.batch_size = _parse_args.batch_size
        self.dataset_scale = 0.8
        self.learn_rate = _parse_args.learn_rate

        # MLP
        input_size = 1280 + 768  # ESM-2模型的输出大小，并展平
        hidden_sizes = [int(item) for item in _parse_args.hidden_sizes.split(',')]  # 隐藏层大小列表
        output_size = 1  # MLP的输出大小
        dropout_prob = _parse_args.dropout_prob  # 被丢弃的概率
        self.model = MLP(input_size, hidden_sizes, output_size, dropout_prob)

        # to cuda
        is_cuda = _parse_args.cuda
        if is_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        if is_cuda and torch.cuda.device_count() > 1:
            print(f"Let's use, {torch.cuda.device_count()}, GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
        else:
            print(f"Let's use, CPU!")

        self.model.to(self.device)

        # 数据集文件
        self.dna_embedding_file = f'{root}/esol/dna_embedding.pkl'
        self.protein_embedding_file = f'{root}/esol/protein_embedding.pkl'
        self.dataset_file = f'{root}/esol/dataset.csv'

        # 2. 保存模型参数
        if not os.path.exists(f'{root}/model'):
            os.makedirs(f'{root}/model')

        # loss, rmse, r2
        self.train_data_file = f'{root}/model/train.data.csv'
        if os.path.exists(self.train_data_file):
            os.remove(self.train_data_file)

        self.predicted_res_data_file = f'{root}/model/predict.data.csv'
        if os.path.exists(self.predicted_res_data_file):
            os.remove(self.predicted_res_data_file)

        # model checkpoint
        self.resume = _parse_args.resume

        # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
        self.last_rmse = sys.maxsize  # 最大值，比这个小 就保存
        self.last_r2 = -sys.maxsize - 1  # 最小值，比这个大 就保存
        self.checkpoint_pt = f'{root}/model/checkpoint.pt'

        # 3. 日志操作
        self.log_file = f'{root}/model/log.txt'
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def log(self, msg, is_print=True):
        with open(self.log_file, 'a', encoding='utf-8') as w:
            w.write(f'{msg}\n')
            if is_print:
                print(msg)

    def append_train_data(self, data):
        with open(self.train_data_file, 'a', encoding='utf-8') as w:
            strs = ''
            for item in data:
                strs += f'{item:.4f},'
            w.write(f'{strs[:-1]}\n')
            return strs[:-1]

    def write_predicted_res(self, predict, target):
        with open(self.predicted_res_data_file, 'a', encoding='utf-8') as w:
            for i in range(len(predict)):
                w.write(f'{predict[i]:.4f},{target[i]:.4f}\n')

    def _set_seed(self):
        seed = self.seed
        random.seed(seed)  # Python的随机性
        np.random.seed(seed)  # numpy的随机性
        os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现

        torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
        torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子

        # 这俩严重降低效率
        # torch.backends.cudnn.deterministic = True  # 选择确定性算法
        # torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False


class Train(Args):
    def __init__(self, _parse_args):
        super().__init__(_parse_args)

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

        # 构建 dataset
        seq_dataset = SequenceDataset(self.dataset_file, self.dna_embedding_file, self.protein_embedding_file)

        # 划分数据集
        train_size = int(self.dataset_scale * len(seq_dataset))
        train_dataset, test_dataset = random_split(seq_dataset, [train_size, len(seq_dataset) - train_size])

        # 加载数据集
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # 断点续训，恢复参数
        if self.resume and os.path.exists(self.checkpoint_pt):
            checkpoint = torch.load(self.checkpoint_pt)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.curr_epoch = checkpoint['epoch'] + 1
            self.last_rmse = checkpoint['rmse']
            self.last_r2 = checkpoint['r2']

            self.log(f'Info: resume checkpoint')

    def run(self):
        self.log(f'Info: start training', is_print=False)
        bar = tqdm.tqdm(total=(self.num_epochs - self.curr_epoch))
        for epoch in range(self.curr_epoch, self.num_epochs):
            # loss, rmse, r2
            train_result = self.train_model(epoch)
            test_result = self.evaluate_model(epoch)

            # 保存数据 后续画图
            info = self.append_train_data(train_result + test_result)

            # 保存模型训练参数
            rmse, r2 = test_result[1], test_result[2]
            if rmse < self.last_rmse or r2 > self.last_r2:
                self.log(f'Best {info}', is_print=False)

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'rmse': self.last_rmse,
                    'r2': self.last_r2
                }

                torch.save(checkpoint, self.checkpoint_pt)

            bar.update(1)
            bar.set_description(f'rmse: {rmse:.3f}, r2: {r2:.3f}')

        bar.close()

    def train_model(self, epoch):
        # need args: model, dataloader, criterion, optimizer, num_epochs
        self.model.train()
        loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0

        for step, (inputs, targets) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)

            # print(inputs.shape, targets.shape, outputs.shape)
            # raise

            # tensor[batch_size, 1] -> tensor[batch_size]
            outputs = torch.squeeze(outputs)

            loss = self.criterion(outputs, targets)

            # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
            outputs_np, targets_np = outputs.detach().to('cpu').numpy(), targets.to('cpu').numpy()

            # 计算均方根误差（RMSE）
            rmse = np.sqrt(mean_squared_error(outputs_np, targets_np))

            # 计算决定系数（R^2） 参数顺序是真实标签在前，预测结果在后
            r2 = r2_score(targets_np, outputs_np)

            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            rmse_total += rmse
            r2_total += r2

            step_count += 1

            self.log(
                f"Train Epoch [{epoch + 1}/{self.num_epochs}], Step [{step}] Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}", is_print=False)

        self.log(f"Train Epoch [{epoch + 1}/{self.num_epochs}], Total Loss: {(loss_total / step_count):.4f}, RMSE: {rmse_total / step_count:.4f}, R^2: {r2_total / step_count:.4f}", is_print=False)

        return [loss_total / step_count, rmse_total / step_count, r2_total / step_count]

    def evaluate_model(self, epoch):
        # need: model, dataloader
        self.model.eval()

        with torch.no_grad():
            loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0

            for step, (inputs, targets) in enumerate(self.test_dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                # tensor[2, 1] -> tensor[2]
                outputs = torch.squeeze(outputs)

                loss = self.criterion(outputs, targets)

                # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
                outputs_np, targets_np = outputs.detach().to('cpu').numpy(), targets.to('cpu').numpy()

                # 计算均方根误差（RMSE）
                rmse = np.sqrt(mean_squared_error(outputs_np, targets_np))

                # 计算决定系数（R^2） 参数顺序是真实标签在前，预测结果在后
                r2 = r2_score(targets_np, outputs_np)

                loss_total += loss.item()
                rmse_total += rmse
                r2_total += r2

                step_count += 1

                self.log(
                    f"Evaluate Epoch [{epoch + 1}/{self.num_epochs}], Step [{step}] Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}", is_print=False)

            self.log(
                    f"Evaluate Epoch [{epoch + 1}/{self.num_epochs}], Total Loss: {(loss_total / step_count):.4f}, RMSE: {rmse_total / step_count:.4f}, R^2: {r2_total / step_count:.4f}", is_print=False)

            return [loss_total / step_count, rmse_total / step_count, r2_total / step_count]

    def predict(self):
        if not os.path.exists(self.checkpoint_pt):
            raise ValueError(f'checkpoint path not find')

        checkpoint = torch.load(self.checkpoint_pt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        with torch.no_grad():
            loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0

            for step, (inputs, targets) in enumerate(self.test_dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                # tensor[2, 1] -> tensor[2]
                outputs = torch.squeeze(outputs)

                loss = self.criterion(outputs, targets)

                # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
                outputs_np, targets_np = outputs.detach().to('cpu').numpy(), targets.to('cpu').numpy()
                # print(outputs_np.shape, targets_np.shape)
                # raise
                self.write_predicted_res(outputs_np, targets_np)

                # 计算均方根误差（RMSE）
                rmse = np.sqrt(mean_squared_error(outputs_np, targets_np))

                # 计算决定系数（R^2） 参数顺序是真实标签在前，预测结果在后
                r2 = r2_score(targets_np, outputs_np)

                loss_total += loss.item()
                rmse_total += rmse
                r2_total += r2

                step_count += 1

                print(f'{step}, {loss.item()}, {rmse}, {r2}')

            self.log(f'predict: {(loss_total / step_count):.4f}, {(rmse_total / step_count):.4f}, {(r2_total / step_count):.4f}')
            with open(self.predicted_res_data_file, 'a', encoding='utf-8') as w:
                w.write(f'predict: {(loss_total / step_count):.4f}, {(rmse_total / step_count):.4f}, {(r2_total / step_count):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DSRP args info")

    parser.add_argument('-ne', '--num_epochs', type=int, default=500)

    parser.add_argument('-hs', '--hidden_sizes', type=str, default='512,128,32')
    parser.add_argument('-dp', '--dropout_prob', type=float, default=0.1, help='MLP 被丢弃的概率')

    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--learn_rate', type=float, default=0.001)

    parser.add_argument('-c', '--cuda', action='store_true', help='是否开启 cuda')
    parser.add_argument('-re', '--resume', action='store_true', help='是否断点续训')
    parser.add_argument('-p', '--predict', action='store_true', help='预测数据')

    parse_args = parser.parse_args()
    print(parse_args)

    train = Train(parse_args)
    if not parse_args.predict:
        train.run()
    else:
        train.predict()

    print('ok')
