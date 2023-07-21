import os.path

from torch.utils.data import Dataset, DataLoader
from DPESol.args import dataset_file


class SequenceDataset(Dataset):
    def __init__(self, esol_file):
        self.source_file = esol_file

        self.gene_name_list = []
        self.gene_solubility_dict = {}
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

                self.gene_name_list.append(gene_name)
                self.gene_solubility_dict[gene_name] = float(solubility)
                self.gene_seq_dict[gene_name] = [nucle_seq, acid_seq]

    def __getitem__(self, index):
        gene_name = self.gene_name_list[index]

        # return x, y
        return self.gene_seq_dict[gene_name], self.gene_solubility_dict[gene_name]

    def __len__(self):
        return len(self.gene_name_list)


if __name__ == '__main__':
    seq_dataset = SequenceDataset(dataset_file)
    print(len(seq_dataset))

    # 数据加载，每批次 4 个
    train_loader = DataLoader(dataset=seq_dataset, batch_size=4, shuffle=True)

    #
    for step, (b_x, b_y) in enumerate(train_loader):
        print(step)
        print(b_x)
        print(b_y)
        break

