model
- seq: [n, 1174], seq_max_len=1174
- esm-2: 输入 [('seq_name', 'protein_seq'')], 特征提取大小 [batch_size, num_tokens, embed_dim]: num_tokens=batch_seq_max_len+2 embed_dim=1280
- mpl: 输入 [batch_size, num_tokens*embed_dim]，输出 [batch_size, 1]
  - 现在是 从esm-2提取的特征 [batch_size, num_tokens, embed_dim]， 由于 num_tokens 是每一个批次序列的最大长度+2，所以导致 MPL 输入不确定
  - 打算将得到的特征补齐到 [batch_size, num_tokens=1176, embed_dim=1280]，MPL 输入就有150万+，但参数太多
  - 所以打算 现将 [batch_size, num_tokens, embed_dim=1280] MPL到 [batch_size, num_tokens, embed_dim=10]，之后展开
  - 再 经过MPL: 输入 [batch_size, 1176*10], 输出 [batch_size, 1]