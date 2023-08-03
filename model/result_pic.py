import matplotlib.pyplot as plt
import numpy as np

from DPESol.args import root


def res_pic(train, test, tag):
    x = range(len(train_loss))
    plt.scatter(x, train, s=2, label='train')
    plt.scatter(x, test, s=4, label='test')

    plt.legend()

    plt.title(f'Change of {tag} value  per epoch')
    plt.xlabel('epoch')
    plt.ylabel(f'{tag}')

    plt.savefig(f'{root}/model/{tag}.png')

    plt.show()


# get data
res_file = f'{root}/model/train.data.csv'

train_loss, test_loss = [], []
train_rmse, test_rmse = [], []
train_r2, test_r2 = [], []

with open(res_file, 'r', encoding='utf-8') as r:
    lines = r.readlines()
    for line in lines:
        line = line.strip()

        if not line:
            break

        # 分割数据
        line_list = line.split(',')
        temp = []
        for item in line_list:
            temp.append(round(float(item), 3))

        line_list = temp

        train_loss.append(line_list[0])
        train_rmse.append(line_list[1])
        train_r2.append(line_list[2])

        test_loss.append(line_list[3])
        test_rmse.append(line_list[4])
        test_r2.append(line_list[5])

res_pic(train_loss, test_loss, 'loss')
res_pic(train_rmse, test_rmse, 'rmse')
res_pic(train_r2, test_r2, 'R^2')




