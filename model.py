import math
import random

import torch
import torch.nn as nn


# 用于key_padding_mask
def get_key_padding_mask(tokens):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == 2] = -torch.inf
    return key_padding_mask


# 原版transoformerPE
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        # 将x和positional encoding相加
        # PE是一个精心设计的算法，不需要学习
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class Transformer_copy(nn.Module):
    def __init__(self, d_model=128):
        super(Transformer_copy, self).__init__()
        # 定义词向量，词典数为10。我们不预测两位小数。
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=128)
        # Transformer
        self.transformer = nn.Transformer(d_model=128, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True)
        # PE
        self.positional_encoding = PositionalEncoding(d_model)
        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.linear = nn.Linear(128, 10)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = get_key_padding_mask(src)
        tgt_key_padding_mask = get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # 到这一步就可以输出了，因为在eval过程中，我们不需要
        #
        return self.linear(out)
