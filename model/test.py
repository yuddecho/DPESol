import torch
import torch.nn.functional as F

# 假设有一个形状为 [3, 2] 的张量
tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])

# 设置目标形状 [3, 6]
target_shape = (3, 6)

# 计算需要在每个维度上扩展的数量
pad_dims = [max(0, target_shape[i] - tensor.size(i)) for i in range(len(target_shape))]

# 使用 pad 函数进行扩展和填充，默认填充值为0
expanded_tensor = F.pad(tensor, (0, pad_dims[1], 0, pad_dims[0]), value=0)

# 查看变换后的张量形状
print("原始张量形状：", tensor.size())                # 输出：torch.Size([3, 2])
print("变换后的张量形状：", expanded_tensor.size())  # 输出：torch.Size([3, 6])
print("变换后的张量：")
print(expanded_tensor)
