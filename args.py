import os
import sys

# 获取当前脚本所在的目录
script_directory = os.path.dirname(os.path.abspath(__file__))

# 获取当前脚本所在目录的父目录的绝对路径
parent_directory = os.path.dirname(script_directory)

root = f'{parent_directory}/protein'
print(f'arg.py root: {root}')


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






