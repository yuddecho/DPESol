import os


# 本地为 测试模式
is_test_mode = True

curr_dir = os.getcwd()
parent_dir, curr_dir_name = os.path.split(curr_dir)
parent_dir, _ = os.path.split(parent_dir)

root = f'{parent_dir}/protein'

# if is_test_mode:
#     # data root dir
#     root = '../protein'
# else:
#     root = '.'

# eSol database
esol_file = f'{root}/esol/all_data.tab'
esol_all_data = f'{root}/esol/esol.csv'

# 处理好的数据
dataset_file = f'{root}/esol/dataset.csv'

# E.coli K-12 info file
nucleotide_seq = f'{root}/k12/sequence.txt'
protein_seq = f'{root}/k12/sequence (1).txt'

# 蛋白质序列最大长度
protein_seq_max_len = 1174



