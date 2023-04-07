import math
import random

import torch
import torch.nn as nn

bos = 0
eos = 1
pad = 2


def generate_random_batch(batch_size, max_length=16):
    src = []
    for i in range(batch_size):
        # 随机生成句子长度
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇，并在开头和结尾增加<bos>和<eos>
        random_nums = [bos] + [random.randint(3, 9) for _ in range(random_len)] + [eos]
        # 如果句子长度不足max_length，进行填充
        random_nums = random_nums + [pad] * (max_length - random_len - 2)
        src.append(random_nums)
    src = torch.LongTensor(src)
    # tgt不要最后一个token，因为无论当tgt输入，无论是pad还是eos，最后一个字符的预测值都是无意义的。
    tgt = src[:, :-1]
    # tgt_y不要第一个的token，因为第一个token是bos，这个字符100%存在，loss计算过程中不考虑该字符。
    tgt_y = src[:, 1:]
    # 计算tgt_y，即要预测的有效token的数量
    n_tokens = (tgt_y != pad).sum()

    # 这里的n_tokens指的是我们要预测的tgt_y中有多少有效的token，后面计算loss要用
    return src, tgt, tgt_y, n_tokens


if __name__ == '__main__':
    src, tgt, tgt_y, n_tokens = generate_random_batch(2)
    print('src=', src)
    print('tgt=', tgt)
    print('tgt_y=', tgt_y)
    print('n_tokens=', n_tokens)
