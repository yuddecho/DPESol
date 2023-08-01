import pickle

from DPESol.args import esol_file, nucleotide_seq, protein_seq, esol_all_data, root

# 7.18 联合 eSol 和 从 NCBI 上 K12 的东西，目前还有一部分未找到，需要继续
# 7.26 从总文件中筛出来的数据进行匹配 -> 发现原因，基因名称部分取至 gene_synonym; 且从 .gb 中提取信息是有效的，因为下载文件没有gene_synonym信息
# 有一部分是在 gene 不在 CDS 中，这部分有基因序列，没有蛋白质序列,

# gen_name: solubility
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

        # gene_name 换成 ECK——number
        eck_num = row_list[1]
        gene_name_k_12 += '-' + eck_num

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
            esol_dict[gene_name_k_12] = solubility
        else:
            # 是有重复的 暂不处理 dcuD
            print(f'Info: processing.py {gene_name_k_12} already exists.')

print(f'esol key-solu: {len(esol_dict.keys())}')

# 2. 读取 .gb 并匹配
e_coli_k12_fasta, e_coli_k12_pkl = f'{root}/k12/e.coli.k12.fasta', f'{root}/k12/e.coli.k12.pkl'

# pkl dict: gene : [nucle_seq, acid_seq]
e_coli_k12_dict = {}
with open(e_coli_k12_pkl, 'rb') as r:
    e_coli_k12_dict = pickle.load(r)

e_coli_k12_dict_keys = list(e_coli_k12_dict.keys())

# 3. 写入结果文件
count = 0
is_taa_tag_tga = 0
with open(esol_all_data, 'w', encoding='utf-8') as w:
    # 特例
    special_case = {'ECK1135': 'ymfN-ECK1135', 'ECK2108': 'pinE-ECK1144-pin', 'gst': 'gstA-ECK1631-gst', 'ptr': 'ptrA-ECK2817-ptr', 'ygaD': '', 'spr': '', 'ade': '', 'aceK': ''}
    keys = esol_dict.keys()
    for key in keys:
        solubility = esol_dict.get(key)

        # 额外调试添加
        key = key.split('-')
        _gene_name, _eck_num = key[0], key[1]

        # 先采用 eck num 来查找
        look_dog = False
        e_coli_k12_dict_key = ''

        found_strings = [s for s in e_coli_k12_dict_keys if _eck_num in s]

        if len(found_strings) == 1:
            e_coli_k12_dict_key = found_strings[0]
        else:
            look_dog = True

        # 再使用 gene name 查找
        if look_dog:
            found_strings = [s for s in e_coli_k12_dict_keys if _gene_name in s]

            if len(found_strings) == 1:
                e_coli_k12_dict_key = found_strings[0]
            else:
                if len(found_strings) == 0:
                    print(_gene_name, _eck_num, found_strings, '0')
                    continue

                if len(found_strings) > 1:
                    # 优先匹配 /gene
                    temp = _gene_name + '-'
                    res_strings = [item for item in found_strings if temp == item[:len(key)+1]]
                    if len(res_strings) == 1:
                        e_coli_k12_dict_key = res_strings[0]
                    else:
                        # 其次匹配 /gene_synonym
                        temp = '-' + _gene_name + '-'
                        res_strings = [item for item in found_strings if temp in item]
                        if len(res_strings) == 1:
                            e_coli_k12_dict_key = res_strings[0]
                        else:
                            print(_gene_name, _eck_num, found_strings, 'no eck,gene-,-gene-')
                            continue

        seq_value = e_coli_k12_dict[e_coli_k12_dict_key]

        if '*' in seq_value[1]:
            is_taa_tag_tga += 1
            continue

        # write
        strs = f'{_gene_name}-{_eck_num},{solubility}'
        for item in seq_value:
            strs += f',{item}'

        w.write(f'{strs}\n')

        count += 1

# 2618, 2613, 5
print(f'total: {len(esol_dict.keys())}, write: {count}, isTAA: {is_taa_tag_tga}, no write: {len(esol_dict.keys()) - count}')



