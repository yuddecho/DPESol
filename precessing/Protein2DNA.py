def reverse_translate(protein_sequence):
    genetic_code = {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'N': ['AAT', 'AAC'],
        'D': ['GAT', 'GAC'],
        'C': ['TGT', 'TGC'],
        'Q': ['CAA', 'CAG'],
        'E': ['GAA', 'GAG'],
        'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAT', 'CAC'],
        'I': ['ATT', 'ATC', 'ATA'],
        'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'K': ['AAA', 'AAG'],
        'M': ['ATG'],
        'F': ['TTT', 'TTC'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'W': ['TGG'],
        'Y': ['TAT', 'TAC'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        '*': ['TAA', 'TAG', 'TGA']  # 终止密码子
    }

    dna_sequence_total = 1
    for amino_acid in protein_sequence:
        codons = genetic_code.get(amino_acid, [''])  # 获取对应的密码子，若未知氨基酸则默认为空字符串
        if len(codons) != 0:
            dna_sequence_total *= len(codons)  # 取第一个密码子，实际应用中可能需要考虑所有可能的密码子
        else:
            print(codons)

    return dna_sequence_total


# 示例使用：
protein_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFAYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
dna_sequence = reverse_translate(protein_sequence)
print(len(protein_sequence), f"{dna_sequence:.2e}")

