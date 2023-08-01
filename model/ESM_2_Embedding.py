"""
url: https://github.com/facebookresearch/esm
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
"""
import os
import pickle
import argparse

import torch
from torch import tensor, nn
from torch.utils.data import Dataset, DataLoader

from DPESol.args import root, dataset_file

"""
    note：使用ESM-2对蛋白序列进行编码
    每个数据编码的对应tensor 都保存到 protein_seq_esm_embedding 文件下
"""


class SequenceDataset(Dataset):
    """
        esol_file: gene_name, nucle_seq, acid_seq
    """
    def __init__(self, esol_file):
        self.source_file = esol_file

        # read data 暂不处理 nucle_seq
        self.gene_name_list = []
        # gene_name: solu
        self.gene_solubility_dict = {}
        # [(gene_name, seq), ] to esm
        self.esm_gene_name_acid_seq_tuple_list = []

        # init
        self._read_file()

        pass

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
                # gene: solu
                self.gene_solubility_dict[_gene_name] = float(solubility)
                # esm
                self.esm_gene_name_acid_seq_tuple_list.append((_gene_name, acid_seq))

    def __getitem__(self, index):
        _gene_name = self.gene_name_list[index]

        # return x, y
        # return self.gene_seq_dict[gene_name], self.gene_solubility_dict[gene_name]
        return self.esm_gene_name_acid_seq_tuple_list[index], self.gene_solubility_dict[_gene_name]

    def __len__(self):
        return len(self.gene_name_list)


def dataset_collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 进行自定义处理，比如转换成张量等
    # data = tensor(data)
    labels = tensor(labels)

    return data, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DPESol args.")

    parser.add_argument('-bs', '--batch_size', type=int, default=2)

    # 输入 --flag 的时候，才会触发 action 对应值, 这里默认 false
    parser.add_argument('-c', '--cuda',  action='store_true', help='是否开启 cuda')

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
    seq_data_loader = DataLoader(dataset=seq_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_collate_fn)

    # esm-2
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # to gpu
    if is_cuda and torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    # embedding
    embedding_res_dict = {}
    for step, (inputs, _) in enumerate(seq_data_loader):
        print(f'=== {step + 1}/{len(seq_data_loader)} ===')

        # 序列转换
        batch_labels, batch_strs, batch_tokens = batch_converter(inputs)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # to gpu
        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        # torch.Size([batch_size, num_tokens, 1280])
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for (gene_name, _), (i, tokens_len) in zip(inputs, enumerate(batch_lens)):
            # torch.Size([batch_size, 1280])
            sequence_representations = token_representations[i, 1: tokens_len - 1].mean(0)

            if embedding_res_dict.get(gene_name, None) is None:
                embedding_res_dict[gene_name] = sequence_representations
            else:
                print(f'Warning: embedding dict -> {gene_name} is exists')

            # print(embedding_res_dict)
            # temp = embedding_res_dict[gene_name]
            # print(temp.shape)
            # raise

            # 改变维度大小 torch.Size([batch_size, num_tokens, 1280]) -> torch.Size([batch_size, num_tokens * 1280])
        # protein_sequence_embedding = protein_sequence_embedding.contiguous().view(protein_sequence_embedding.size(0), -1)

    embedding_res_file = f'{root}/esol/protein_embedding.pkl'
    with open(embedding_res_file, 'wb') as w:
        pickle.dump(embedding_res_dict, w)

    print(f'ok')








