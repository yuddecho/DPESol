"""
规范下载 sequence.txt 规范成 fasta 文件
"""

from DPESol.args import root
import os


# path
if os.path.exists(root):
    print(root)
else:
    print(root)
    raise ValueError('路径不存在')


def format_seq(source, target):
    total_count = 0
    with open(target, 'w', encoding='utf-8') as w:
        with open(source, 'r', encoding='utf-8') as r:
            rows = r.readlines()

            index, index_max = 0, len(rows)
            while True:
                if index >= index_max:
                    break

                row = rows[index]
                row = row.strip()

                if row == '':
                    break

                if row[0] == '>':
                    fasta_head = row

                    # 找序列
                    seq_info = ''
                    index += 1
                    while True:
                        if index >= index_max:
                            break

                        row = rows[index]
                        row = row.strip()

                        if row == '':
                            break

                        if row[0] == '>':
                            break

                        seq_info += row

                        index += 1

                    w.write(f'{fasta_head}\n')
                    w.write(f'{seq_info}\n')

                    total_count += 1

                    continue

                index += 1

    return total_count


nucle_seq_file, acid_seq_file = f'{root}/k12/sequence.txt', f'{root}/k12/sequence (1).txt'
nucle_seq_fasta, acid_seq_fasta = f'{root}/k12/nucle.fasta', f'{root}/k12/acid.fasta'

print(format_seq(nucle_seq_file, nucle_seq_fasta))
print(format_seq(acid_seq_file, acid_seq_fasta))

