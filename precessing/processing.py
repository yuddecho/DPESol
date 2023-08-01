from DPESol.args import esol_file, nucleotide_seq, protein_seq, esol_all_data

# 7.18 联合 eSol 和 从 NCBI 上 K12 下载的序列信息，目前还有一部分未找到，需要继续

# gen_name: [solubility, nucleotide, protein]
esol_dict = {}

# 1. 解析 eSol database file
with open(esol_file, 'r', encoding='utf-8') as r:
    # 丢掉第一行 表头
    r.readline()

    rows = r.readlines()
    for row in rows:
        row = row.strip()
        if not row:
            break

        row_list = row.split('	')

        # get gene name 这俩名字不一定相等，所以不判别了
        gene_name_k_12 = row_list[3]
        # gene_name_k_12, locus_name_k_12 = row_list[3], row_list[4]
        # if gene_name_k_12 != locus_name_k_12:
        #     print(f'Info: processing.py {gene_name_k_12} != {locus_name_k_12}.')
        #     continue

        # get solubility
        solubility = row_list[6]
        if solubility == '':
            # 没有数值 过滤掉
            continue
        else:
            solubility = float(solubility) * 0.01

            # 规整到 [0, 1]
            if solubility > 1:
                solubility = 1.0

        # add dict
        if esol_dict.get(gene_name_k_12, None) is None:
            esol_dict[gene_name_k_12] = [solubility]
        else:
            # 是有重复的 暂不处理 dcuD
            print(f'Info: processing.py {gene_name_k_12} already exists.')

print(f'esol key-solu: {len(esol_dict.keys())}')


# 2.解析 nucleotide and protein seq
def get_sequence(info_dict, seq_info_file):
    with open(seq_info_file, 'r', encoding='utf-8') as _r:
        current_row = ''
        look_dog = False

        while True:
            gene_name, sequence = '', ''

            if current_row == '':
                current_row = _r.readline()
                current_row = current_row.strip()
                if not current_row:
                    break

            if current_row[0] != '>':
                print(f'Error: processing.py not fasta name: {current_row}.')
                raise

            # fasta info
            gene_label_first_index = current_row.find('[gene=')
            gene_label_second_index = current_row.find(']')
            gene_name = current_row[gene_label_first_index + 6: gene_label_second_index]

            # get sequence
            while True:
                current_row = _r.readline()
                current_row = current_row.strip()
                if not current_row:
                    look_dog = True
                    break

                if current_row[0] == '>':
                    break

                sequence += current_row

            # add data
            info_value = info_dict.get(gene_name, None)
            if info_value is not None:
                info_value.append(sequence)
                info_dict[gene_name] = info_value

            if look_dog:
                break

    return info_dict


esol_dict = get_sequence(esol_dict, nucleotide_seq)
# 3173 - 7 3166
print(f'esol key-solu-nucl: {len(esol_dict.keys())}')

esol_dict = get_sequence(esol_dict, protein_seq)
print(f'esol key-solu-nucl-prot: {len(esol_dict.keys())}')

# 3. 写入结果文件
count = 0
with open(esol_all_data, 'w', encoding='utf-8') as w:
    keys = esol_dict.keys()
    for key in keys:
        value = esol_dict.get(key)
        if len(value) != 3 and len(value) != 1:
            print(f'Warning: processing.py esol_dict.get(key) len not 1 or 3: {key}.')
            continue

        # 先过滤找不到的数据
        if len(value) == 1:
            continue

        # write
        strs = f'{key}'
        for item in value:
            strs += f',{item}'

        w.write(f'{strs}\n')

        count += 1

# 2618, 2613, 5
print(f'total: {len(esol_dict.keys())}, write: {count}, other: {len(esol_dict.keys()) - count}')



