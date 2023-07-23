import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import mean_squared_error, r2_score

from DPESol.args import dataset_file, root
from dataset import SequenceDataset, dataset_collate_fn


class Args:
    def __init__(self):
        # 设置参数，保存模型数据，保存日志，设置随机数种子

        # 1. 设置模型参数
        self.curr_epoch = 0
        self.num_epochs = 600
        self.batch_size = 16
        self.dataset_scale = 0.8
        self.learn_rate = 0.001

        # 数据集文件
        self.dataset_file = dataset_file

        # 2. 保存模型参数
        # loss, rmse, r2
        self.train_data_file = f'{root}/model/train.data.csv'
        if os.path.exists(self.train_data_file):
            os.remove(self.train_data_file)

        # model checkpoint
        self.resume = True
        # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
        self.last_rmse = sys.maxsize  # 最大值，比这个小 就保存
        self.last_r2 = -sys.maxsize - 1  # 最小值，比这个大 就保存
        self.checkpoint_pt = f'{root}/model/checkpoint.pt'

        # 3. 日志操作
        self.log_file = f'{root}/model/log.txt'
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # 4. 设置随机数种子
        seed = 2023
        random.seed(seed)  # Python的随机性
        np.random.seed(seed)  # numpy的随机性
        os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现

        torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
        torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子

        # 这俩严重降低效率
        # torch.backends.cudnn.deterministic = True  # 选择确定性算法
        # torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False

    def log(self, msg, is_print=True):
        with open(self.log_file, 'a', encoding='utf-8') as w:
            w.write(f'{msg}\n')
            if is_print:
                print(msg)

    def append_train_data(self, data):
        with open(self.train_data_file, 'a', encoding='utf-8') as w:
            strs = ''
            for item in data:
                strs += f'{item},'
            w.write(f'{strs[:-1]}\n')
            return strs[:-1]

    def print(self):
        pass


class Train(Args):
    def __init__(self):
        super().__init__()

        # device
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 定义 model

        # ESM-2
        self.esm_model, self.esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval()
        self.log(f'Info: esm2_t33_650M_UR50D load finish', True)

        self.model = DPESol(self.esm_model)
        self.log(f'\n{self.model}\n', False)

        # 模型
        if torch.cuda.device_count() > 1:
            self.log(f"Let's use, {torch.cuda.device_count()}, GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        # print(model)

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)

        # 构建 dataset
        seq_dataset = SequenceDataset(self.dataset_file)

        # 划分数据集
        train_size = int(self.dataset_scale * len(seq_dataset))
        train_dataset, test_dataset = random_split(seq_dataset, [train_size, len(seq_dataset) - train_size])

        # 加载数据集
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                           collate_fn=dataset_collate_fn)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True,
                                          collate_fn=dataset_collate_fn)

        # 断点续训，恢复参数
        if self.resume and os.path.exists(self.checkpoint_pt):
            checkpoint = torch.load(self.checkpoint_pt)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.curr_epoch = checkpoint['epoch'] + 1
            self.last_rmse = checkpoint['rmse']
            self.last_r2 = checkpoint['r2']

            self.log(f'Info: resume checkpoint', True)

    def run(self):
        self.log(f'Info: start training', True)
        for epoch in range(self.curr_epoch, self.num_epochs):
            # loss, rmse, r2
            train_result = self.train_model(epoch)
            test_result = self.evaluate_model(epoch)

            # 保存数据 后续画图
            info = self.append_train_data(train_result + test_result)

            # 保存模型训练参数
            rmse, r2 = (train_result[1] + test_result[1]) / 2.0, (train_result[2] + test_result[2]) / 2.0

            # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
            if rmse < self.last_rmse or r2 > self.last_r2:
                self.log(f'Best {info}')

                # 保存模型参数
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'rmse': self.last_rmse,
                    'r2': self.last_r2
                }

                torch.save(checkpoint, self.checkpoint_pt)

    def train_model(self, epoch):
        # need args: model, dataloader, criterion, optimizer, num_epochs
        self.model.train()
        loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0

        for step, (inputs, targets) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            # 序列转换
            batch_labels, batch_strs, batch_tokens = self.batch_converter(inputs)
            # batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            batch_tokens = batch_tokens.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(batch_tokens)

            # tensor[2, 1] -> tensor[2]
            outputs = torch.squeeze(outputs)
            # print(outputs.shape, targets.shape)
            # raise

            # output: [batch_size, max_tokens_len, 1], target: [batch_size, 1]
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
                f"Train Epoch [{epoch + 1}/{self.num_epochs}], Step [{step}] Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

        self.log(f"Train Epoch [{epoch + 1}/{self.num_epochs}], Total Loss: {(loss_total / step_count):.4f}, RMSE: {rmse_total / step_count:.4f}, R^2: {r2_total / step_count:.4f}")

        return [loss_total / step_count, rmse_total / step_count, r2_total / step_count]

    def evaluate_model(self, epoch):
        # need: model, dataloader
        self.model.eval()

        with torch.no_grad():
            loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0

            for step, (inputs, targets) in enumerate(self.test_dataloader):
                # 序列转换
                batch_labels, batch_strs, batch_tokens = self.batch_converter(inputs)
                # batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

                batch_tokens = batch_tokens.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(batch_tokens)

                # tensor[2, 1] -> tensor[2]
                outputs = torch.squeeze(outputs)
                # print(outputs.shape, targets.shape)
                # raise

                # output: [batch_size, max_tokens_len, 1], target: [batch_size, 1]
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
                    f"Evaluate Epoch [{epoch + 1}/{self.num_epochs}], Step [{step}] Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

            self.log(
                    f"Evaluate Epoch [{epoch + 1}/{self.num_epochs}], Total Loss: {(loss_total / step_count):.4f}, RMSE: {rmse_total / step_count:.4f}, R^2: {r2_total / step_count:.4f}")

            return [loss_total / step_count, rmse_total / step_count, r2_total / step_count]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="DPESol args.")

    # 输入 --flag 的时候，才会触发 action 对应值
    parser.add_argument('--web_mode', action='store_true')

    args = parser.parse_args()

    web_mode = args.web_mode
    print(f'web_mode: {web_mode}')

    if web_mode:
        from DPESol.model.model_web import DPESol
    else:
        from DPESol.model.model import DPESol

    train = Train()
    train.run()
