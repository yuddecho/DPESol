import pickle

from DPESol.args import root

import os
import re

# path
if os.path.exists(root):
    print(root)
else:
    print(root)
    raise ValueError('路径不存在')

# 处理 k12 gb 文件


def translate_dna_to_protein(dna_sequence):
    # 定义氨基酸密码子与对应氨基酸的字典
    codon_table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
        'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
    }

    protein_sequence = ''
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i+3].upper()
        if codon in codon_table:
            protein_sequence += codon_table[codon]
        else:
            protein_sequence += 'X'  # 未知的密码子使用 X 表示
    return protein_sequence


# gbk file
e_coli_k12_gbk = f'{root}/k12/sequence.gb'
if os.path.exists(e_coli_k12_gbk):
    print(e_coli_k12_gbk)

e_coli_k12_fasta, e_coli_k12_pkl = f'{root}/k12/e.coli.k12.fasta', f'{root}/k12/e.coli.k12.pkl'

count = 0
with open(e_coli_k12_gbk, 'r', encoding='utf-8') as r:
    rows = r.readlines()

    # gene : [index1, index2, acid_seq]
    k12_dict = {}

    # 有效信息 gene, cds 在 [145, 112670]
    index, index_max = 145, 112670
    gene_info = []
    mutli_location_count, no_translation_count = 0, 0
    look_dog, look_dog_count = False, 0
    while True:
        # 遍历完成 112670 有效信息也会被读取
        if index > index_max:
            break

        # 读取一行数据
        row = rows[index]
        row = row.strip()

        # 从 gene 下 获取 location 和 gene_name
        if row[:4] == 'gene':
            if look_dog:
                # 说明没有遇到 CDS 段
                look_dog_count += 1

            # 遇到 gene 段
            look_dog = True

            # 使用正则表达式匹配数字 找到基因序列对应下标
            locations = re.findall(r'\d+', row)

            if len(locations) != 2:
                print(row)
                mutli_location_count += 1
                continue

            # 继续读取 寻找完整 gene name
            gene_name = ''
            while True:
                index += 1

                # 遍历完成
                if index > index_max:
                    break

                # 读取一行数据
                row = rows[index]
                row = row.strip()

                # /gene="chpS"
                if row[:6] == '/gene=':
                    gene_name += row[7:-1]

                # /gene_synonym="chpBI"
                if row[:13] == '/gene_synonym':
                    gene_name += '-' + row[15: -1]

                # 遇到 CDS 段 撤
                if row[:3] == 'CDS':
                    break

                # 遇到新 gene 段 撤
                if row[:4] == 'gene':
                    break

            if gene_name == '':
                print(row)
                raise ValueError('没有找到 gene name')

            # 暂存信息
            gene_info = [gene_name] + locations

            # 先存一部分
            locations.append('')
            k12_dict[gene_name] = locations

        # 从 CDS 下 获取 /translation 信息
        if row[:3] == 'CDS':
            # CDS 接 CDS，剔除
            if not look_dog:
                # 找到下一个标签入口
                while True:
                    index += 1

                    # 遍历完成
                    if index > index_max:
                        break

                    # 读取一行数据
                    row = rows[index]
                    row = row.strip()

                    if row[:4] == 'gene':
                        break

                    if row[:3] == 'CDS':
                        break

                continue

            # 遇到 新 CDS 段
            look_dog = False

            # 继续读取 寻找完整 gene name
            translation = ''
            while True:
                index += 1

                # 遍历完成
                if index > index_max:
                    break

                # 读取一行数据
                row = rows[index]
                row = row.strip()

                # /translation
                if row[:12] == '/translation':
                    translation = row
                    # 继续读取
                    while True:
                        index += 1

                        # 遍历完成
                        if index > index_max:
                            break

                        # 读取一行数据
                        row = rows[index]
                        row = row.strip()

                        if row[:4] == 'gene':
                            break

                        if row[:3] == 'CDS':
                            break

                        translation += row

                # 遇到 gene 段 撤
                if row[:4] == 'gene':
                    break

                if row[:3] == 'CDS':
                    break

            if translation != '':
                translation = translation[14: -1]
            else:
                no_translation_count += 1

            # load info
            gene_name, locations = gene_info[0], gene_info[1:]

            # 重新赋值
            k12_dict[gene_name] = locations + [translation]

            # count += 1
            # if count == 10:
            #     print(k12_dict)
            #     raise

    print('k12 dict len:', len(k12_dict.keys()), '; no CDS:', look_dog_count, '; mutli location:', mutli_location_count, '; no translation', no_translation_count)

    # 有效信息 nucle seq 在 [112672, 190032]
    index, index_max = 112672, 190032
    all_nucle_seq = ''
    while True:
        # 遍历完成 190032 有效信息也会被读取
        if index > index_max:
            break

        # 读取一行数据
        row = rows[index]
        row = row.strip()

        first_space_index = row.find(' ')
        row = row[first_space_index + 1:]
        row = row.replace(' ', '')

        all_nucle_seq += row

        index += 1

    print(len(all_nucle_seq))
    # print(all_nucle_seq[:60])
    # print(all_nucle_seq[-52:])

    all_nucle_seq = all_nucle_seq.upper()

    # gene : [nucle_seq, acid_seq]
    k12_gene_acid_nucle_dict = {}
    no_translation_count, no_taa_tag_tga = 0, 0
    with open(e_coli_k12_fasta, 'w', encoding='utf-8') as w:
        for key in k12_dict.keys():
            _gene_name = key

            # 下标从 0 开始，原始计数从 1 开始
            index_1, index_2 = int(k12_dict[_gene_name][0]) - 1, int(k12_dict[_gene_name][1])
            _acid_seq = k12_dict[_gene_name][2]

            _nucle_seq = all_nucle_seq[index_1: index_2]

            if _acid_seq == '':
                no_translation_count += 1
                _acid_seq = translate_dna_to_protein(_nucle_seq)

                if _acid_seq[-1] == '*' or _acid_seq[-1] == 'X':
                    _acid_seq = _acid_seq[:-1]

            if '*' in _acid_seq:
                no_taa_tag_tga += 1
                # continue

            w.write(f'>{_gene_name}\n')
            w.write(f'{_nucle_seq}\n')
            w.write(f'{_acid_seq}\n')

            k12_gene_acid_nucle_dict[_gene_name] = [_nucle_seq, _acid_seq]

    with open(e_coli_k12_pkl, 'wb') as w:
        pickle.dump(k12_gene_acid_nucle_dict, w)

    print('pkl len: ', len(k12_gene_acid_nucle_dict.keys()), '; No Trans:', no_translation_count, '; No TAA:', no_taa_tag_tga)

# root: /Users/yudd/code/python/echo/protein
# /Users/yudd/code/python/echo/protein
# /Users/yudd/code/python/echo/protein/k12/sequence.gb
# k12 dict len: 4639 ; no CDS: 334 ; mutli location: 0 ; no translation 17
# pkl len:  4639 ; No Trans: 351

# k12 一共 4639 个 gene, 其中 334 个没有 CDS， 有CDS中，有17个没有给翻译序列


