import torch
from transformers import AutoTokenizer, AutoModel

import os
import pickle
import argparse

import torch
from torch import tensor, nn
from torch.utils.data import Dataset, DataLoader

from DPESol.args import root, dataset_file

"""
    note：使用DNABERT-2对蛋白序列进行编码
"""


class SequenceDataset(Dataset):
    """
        esol_file: gene_name, nucle_seq, acid_seq
    """
    def __init__(self, esol_file):
        self.source_file = esol_file

        # read data
        self.gene_name_list = []
        self.nucle_list = []

        # init
        self._read_file()

    def _read_file(self):
        with open(self.source_file, 'r', encoding='utf-8') as r:
            rows = r.readlines()
            for row in rows:
                row = row.strip()
                row_list = row.split(',')

                # get data 暂不处理 nucle_seq
                _gene_name, solubility, nucle_seq, acid_seq = row_list[0], row_list[1], row_list[2], row_list[3]

                # gene name
                self.gene_name_list.append(_gene_name)
                # nucle seq
                self.nucle_list.append(nucle_seq)

    def __getitem__(self, index):
        return self.nucle_list[index], self.gene_name_list[index]

    def __len__(self):
        return len(self.gene_name_list)


def dataset_collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 进行自定义处理，比如转换成张量等
    # data = tensor(data)
    # labels = tensor(labels)

    return data, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DPESol args.")

    parser.add_argument('-bs', '--batch_size', type=int, default=1)

    # 输入 --flag 的时候，才会触发 action 对应值, 这里默认 false
    parser.add_argument('-c', '--cuda', action='store_true', help='是否开启 cuda')

    args = parser.parse_args()
    print(args)

    # args
    is_cuda = args.cuda
    if is_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    batch_size = args.batch_size

    # dataset
    seq_dataset = SequenceDataset(dataset_file)
    seq_data_loader = DataLoader(dataset=seq_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=dataset_collate_fn)

    # https://huggingface.co/zhihan1996/DNABERT-2-117M
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    # to gpu
    if is_cuda and torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    # embedding
    embedding_res_dict = {}
    for step, (seqs, gene_names) in enumerate(seq_data_loader):
        print(f'=== {step + 1}/{len(seq_data_loader)} ===')

        # 序列转换
        # inputs = tokenizer(seqs, return_tensors='pt', padding=True)["input_ids"]
        inputs = tokenizer(seqs, return_tensors='pt')["input_ids"]
        inputs = inputs.to(device)

        with torch.no_grad():
            hidden_states = model(inputs)[0]

        # 找到每个序列表示, 因为 不清楚 hidden_states size 中 第二维度的大小，如果多序列一起输入，会加入padding，在结果中求均值会带来误差
        # 所以默认 batch_size 为 1
        embedding_mean = torch.mean(hidden_states[0], dim=0)

        gene_name = gene_names[0]
        if embedding_res_dict.get(gene_name, None) is None:
            embedding_res_dict[gene_name] = embedding_mean
        else:
            print(f'Warning: embedding dict -> {gene_name} is exists')

    embedding_res_file = f'{root}/esol/dna_embedding.pkl'
    with open(embedding_res_file, 'wb') as w:
        pickle.dump(embedding_res_dict, w)

    print(f'ok')
