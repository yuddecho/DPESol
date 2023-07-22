import os

# 获取根目录绝对路径
curr_dir = os.getcwd()
parent_dir, curr_dir_name = os.path.split(curr_dir)
parent_dir, _ = os.path.split(parent_dir)

root = f'{parent_dir}/protein'


# 1. 数据处理过程，文件路径参数
# eSol database
esol_file = f'{root}/esol/all_data.tab'
esol_all_data = f'{root}/esol/esol.csv'

# 处理好的数据, 直接送入 dataset
dataset_file = f'{root}/esol/dataset.csv'

# E.coli K-12 info file
nucleotide_seq = f'{root}/k12/sequence.txt'
protein_seq = f'{root}/k12/sequence (1).txt'

# 蛋白质序列最大长度
protein_seq_max_len = 1174

if __name__ == '__main__':
    print(root)






