import torch
import torch.nn as nn


class SimpleTextClassifier(nn.Module):
    def __init__(self):
        super(SimpleTextClassifier, self).__init__()

        in_channel = 1280
        out_channel = int(1280 * 1.5)
        kernel_size = 3
        self.con = nn.Sequential(
                # out: (b, out_channel, l)
                nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=int(kernel_size / 2)),
                nn.ReLU(),
                # out: (b, out_channel, l / kernel_size)
                nn.MaxPool1d(kernel_size=kernel_size),

                # out: (b, in_channel, l / kernel_size)
                nn.Conv1d(in_channels=out_channel, out_channels=in_channel, kernel_size=kernel_size),
                nn.ReLU(),

                # out: (b, in_channel)
                nn.MaxPool1d(kernel_size=kernel_size)
            )

    def forward(self, x):
        # x: (batch_size, sequence_length)

        embedded = x.permute(0, 2, 1)  # (batch_size, embedding_dim, sequence_length)

        output = self.con(embedded)  # (batch_size, num_classes)
        output = torch.max(output, dim=2)[0]
        return output


class ProteinEmbeddingCNN(nn.Module):
    def __init__(self):
        super(ProteinEmbeddingCNN, self).__init__()
        # args
        ModuleList_kernel_list = range(2, 10)
        Sequential_channel_list = [1280, 1920, 1280]

        self.convs = nn.ModuleList()
        for kernel_size in ModuleList_kernel_list:
            layers = []
            for channel_index in range(len(Sequential_channel_list) - 1):
                layers.append(nn.Conv1d(in_channels=Sequential_channel_list[channel_index],
                                        out_channels=Sequential_channel_list[channel_index + 1],
                                        kernel_size=kernel_size,
                                        padding=int(kernel_size / 2)))
                layers.append(nn.ReLU())
                if channel_index + 2 != len(Sequential_channel_list):
                    layers.append(nn.MaxPool1d(kernel_size=kernel_size))

            self.convs.append(nn.Sequential(*layers))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # out = [torch.max(conv(x), dim=2)[0] for conv in self.convs]
        out_total = []
        for conv in self.convs:
            print(x.shape)

            out = conv(x)
            print(out.shape)

            out = torch.max(conv(x), dim=2)[0]
            print(out.shape)

            out_total.append(out)

        return out_total


# 创建模型实例
model = ProteinEmbeddingCNN()
print(model)

# 随机生成输入数据并进行前向传播
batch_size = 32
sequence_length = 9
input_data = torch.rand(batch_size, sequence_length, 1280)
output = model(input_data)

print("输出尺寸:", len(output))
