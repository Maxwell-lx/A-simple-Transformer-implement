import math
import random

import torch
import torch.nn as nn
from model import Transformer_copy
from dataset import generate_random_batch

bos = 0
eos = 1
pad = 2

def get_key_padding_mask(tokens):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == pad] = -torch.inf
    return key_padding_mask



if __name__ == '__main__':
    max_length = 16
    model = Transformer_copy()
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    total_loss = 0

    for step in range(2000):
        # 生成数据
        src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size=2, max_length=max_length)
        # 清空梯度
        optimizer.zero_grad()
        # 进行transformer的计算
        out = model(src, tgt)
        """
        计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                (batch_size*词数, 词典大小)。
                而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                除以n_tokens。
        """
        loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens

        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        total_loss += loss

        # 每40次打印一下loss
        if step != 0 and step % 40 == 0:
            print("Step {}, total_loss: {}".format(step, total_loss))
            total_loss = 0

    # eval model
    model = model.eval()
    # 随便定义一个src
    src = torch.LongTensor([[bos, 4, 3, 4, 6, 8, 9, 9, 8, eos, pad, pad]])
    # tgt从<bos>开始，看看能不能重新输出src中的值
    tgt = torch.LongTensor([[bos]])

    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(max_length):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        out = out[:, -1]
        # 找出最大值的index
        y = torch.argmax(out, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)

        # 如果为<eos>，说明预测结束，跳出循环
        if y == eos:
            break
    print(tgt)
