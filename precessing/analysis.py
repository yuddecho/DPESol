from DPESol.args import esol_all_data, dataset_file

import matplotlib.pyplot as plt
import numpy as np

# 分析 序列组成 和 长度分布

# 1. 获取数据
nucleotide_seq_len_dict, acid_seq_len_dict = {}, {}
nucleotide_set, acid_set = set(), set()
data_count = 0

with open(esol_all_data, 'r', encoding='utf-8') as r:
    rows = r.readlines()
    data_count = len(rows)
    for row in rows:
        row = row.strip()
        row_list = row.split(',')

        # get data
        gene_name, nucle_seq, acid_seq = row_list[0], row_list[2], row_list[3]

        # get seq len
        nucle_seq_len, acid_seq_len = len(nucle_seq), len(acid_seq)
        nucleotide_seq_len_dict[nucle_seq_len] = nucleotide_seq_len_dict.get(nucle_seq_len, 0) + 1
        acid_seq_len_dict[acid_seq_len] = acid_seq_len_dict.get(acid_seq_len, 0) + 1

        # get seq set
        nucleotide_set, acid_set = nucleotide_set.union(set(nucle_seq)), acid_set.union(set(acid_seq))

nucle_seq_min_len, acid_seq_min_len = min(list(nucleotide_seq_len_dict.keys())), min(list(acid_seq_len_dict.keys()))
nucle_seq_max_len, acid_seq_max_len = max(list(nucleotide_seq_len_dict.keys())), max(list(acid_seq_len_dict.keys()))

print(f'seq info: {len(nucleotide_set)}: [{nucle_seq_min_len}, {nucle_seq_max_len}] {nucleotide_set}, {len(acid_set)}: [{acid_seq_min_len}, {acid_seq_max_len}] {acid_set}')


# 检查 前 95% 数据
total = data_count
max_seq_num = int(total * 0.99)
print(f'{total}: {max_seq_num}: {total - max_seq_num}')

# 2. 画图
plt.figure(figsize=(14, 6))

plt.subplot(121)

x, y = list(nucleotide_seq_len_dict.keys()), list(nucleotide_seq_len_dict.values())
# x = list(nucleotide_seq_len_dict.keys())
# x.sort()
# x = x[:max_seq_num]
# y = [nucleotide_seq_len_dict[item] for item in x]

plt.scatter(x, y, s=2, label='nucleotide')

# 画 限定线
x.sort()

seq_count, x_max_index = 0, -1
for _x in x:
    seq_count += nucleotide_seq_len_dict[_x]
    if seq_count > max_seq_num:
        x_max_index = _x
        break

print(f'nucleotide max len: {x_max_index}')
plt.plot([x_max_index, x_max_index], [0, 16], label='max_len')


plt.legend()

new_ticks = np.linspace(0, 16, 9)
plt.yticks(new_ticks)

plt.title('nucleotide')
plt.xlabel('seq len')
plt.ylabel('seq num')


# 图二
plt.subplot(122)

x, y = list(acid_seq_len_dict.keys()), list(acid_seq_len_dict.values())
# x = list(acid_seq_len_dict.keys())
# x.sort()
# x = x[:max_seq_num]
# y = [acid_seq_len_dict[item] for item in x]

plt.scatter(x, y, s=2, label='protein')

# 画 限定线
x.sort()
seq_count, x_max_index = 0, -1
for _x in x:
    seq_count += acid_seq_len_dict[_x]
    if seq_count > max_seq_num:
        x_max_index = _x
        break

print(f'acid max len: {x_max_index}')
plt.plot([x_max_index, x_max_index], [0, 16], label='max_len')


plt.legend()

new_ticks = np.linspace(0, 16, 17)
plt.yticks(new_ticks)

plt.title('protein')
plt.xlabel('seq len')
plt.ylabel('seq num')

plt.savefig('res.png')

# plt.show()

# nucleotide max len: 3525
# acid max len: 1174

# 3. 长度筛选 好像不需要
nucle_max_len, acid_max_len = 2745, 914
count = 0
with open(dataset_file, 'w', encoding='utf-8') as w:
    with open(esol_all_data, 'r', encoding='utf-8') as r:
        rows = r.readlines()
        for row in rows:
            row = row.strip()
            row_list = row.split(',')

            # get data
            nucle_seq, acid_seq = row_list[2], row_list[3]

            # if len(nucle_seq) > nucle_max_len or len(acid_seq) > acid_max_len:
            #     continue

            w.write(f'{row}\n')
            count += 1

print(count)
