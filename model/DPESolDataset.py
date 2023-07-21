import os.path

from torch.utils.data import Dataset, DataLoader
from torch import tensor
from DPESol.args import dataset_file


class SequenceDataset(Dataset):
    def __init__(self, esol_file):
        self.source_file = esol_file

        # read data
        self.gene_name_list = []
        # gene_name: solu
        self.gene_solubility_dict = {}
        # [(gene_name, seq), ] to esm
        self.esm_gene_name_seq_tuple_list = []

        self.gene_seq_dict = {}

        # init
        self._read_file()

        pass

    def _read_file(self):
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
                # esm
                self.esm_gene_name_seq_tuple_list.append((gene_name, acid_seq))

                self.gene_seq_dict[gene_name] = [nucle_seq, acid_seq]

    def __getitem__(self, index):
        gene_name = self.gene_name_list[index]

        # return x, y
        # return self.gene_seq_dict[gene_name], self.gene_solubility_dict[gene_name]
        return self.esm_gene_name_seq_tuple_list[index], self.gene_solubility_dict[gene_name]

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
    seq_dataset = SequenceDataset(dataset_file)
    print(len(seq_dataset))

    # 数据加载，每批次 4 个
    train_loader = DataLoader(dataset=seq_dataset, batch_size=2, shuffle=True, collate_fn=dataset_collate_fn)

    #
    for step, (b_x, b_y) in enumerate(train_loader):
        print(step)
        print(b_x)
        print(b_y)
        break

