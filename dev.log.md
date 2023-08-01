7.18
- init
- 处理数据
- 名子参考 -> RPPSP: A Robust and Precise Protein Solubility Predictor by Utilizing Novel Protein Sequence Encoder
- 数据来源：
  - esol/all_data.tab: http://www.tanpaku.org/tp-esol/download.php?lang=ja
  - E.coli K12 info: https://www.ncbi.nlm.nih.gov/nuccore/U00096.3 保存 序列和基因特征
    - 整个文件 sequence.gb
    - DNA序列 sequence.txt
    - 蛋白质序列 sequence (1).txt
    - 基因特征 sequence (2).txt
  - 数据处理
    - 将 gene_name, solubility, nucleotide_seq, acid_seq 组织起来
    - 参考 19 年文章，补充文件里，有26个坏数据；溶解度规整到 [0, 1]
    - 目前数据量只有 2613 个，后续再挖掘别的数据，丢到的部分是没有找到序列的，需要再找
    - 还需要做查看一下，数据分布情况
7.23
- 完成ESM-2蛋白质序列中提取特征，MPL拟合，指标评价，多GPU的，但是显存太小了，跑不下，还是要拆分模型，先用ESM-2提取完特征，在做其他的

7.26
- E.coli k12 一共 4639 个 gene, 其中 334 个没有 CDS， 有CDS中，有17个没有给翻译序列(No Trans: 351), 这部分采用标准密码子对照表进行翻译，翻译后有218条序列中间含有终止密码子
- eSol 中有效数据一共 3173，去除重复 7个，得到 3166个，匹配 k12 得到 3091 条数据
- 如果 不去除终止密码子 3166-3154-12
