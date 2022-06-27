import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from bert4ms.models import BertModel
import mindspore.numpy as mnp 

class ImageEmbedding(nn.Cell):
    def __init__(self, output_size=768):
        super(ImageEmbedding, self).__init__()
        self.output_size = output_size
        self.in_channels = 3
        self.channels = 64
        self.dropout = nn.Dropout(keep_prob=0.5)
        # 112x112x64
        # 56x56x64
        # 56x56x128
        # 28x28x128
        # 14x14x256
        self.simple_cnn = nn.SequentialCell([
            nn.Conv2d(self.in_channels, self.channels, kernel_size=3, stride=2, padding=0, pad_mode='same'),
            nn.BatchNorm2d(self.channels, eps=1e-4, momentum=0.9, gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same"),
            nn.Conv2d(self.channels, self.channels * 2, kernel_size=3, stride=1, padding=0, pad_mode='same'),
            nn.BatchNorm2d(self.channels*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(self.channels * 2, self.channels*4, kernel_size=3, stride=1, padding=0, pad_mode='same'),
            nn.BatchNorm2d(self.channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(self.channels*4, output_size, kernel_size=3, stride=1, padding=0, pad_mode='same')
        ])

    def construct(self, x):
        x = self.simple_cnn(x)
        N = x.shape[0]
        return x.reshape((N, 196, self.output_size))


class Attention(nn.Cell):
    def __init__(self, d=768, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Dense(d, k)
        self.ff_ques = nn.Dense(d, k)
        if dropout:
            self.dropout = nn.Dropout(keep_prob=0.5)
        self.ff_attention = nn.Dense(k, 1)
        self.expand_dims = P.ExpandDims()
        self.softmax = ops.Softmax()

    def construct(self, vi, vq):
        # N * 196 * 512 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq)#.ExpandDims(dim=1)
        hq = self.expand_dims(hq, 1)
        # N * 196 * 512
        tanh = nn.Tanh()
        ha = tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(axis=2)
        pi = self.softmax(ha)
        pi = self.expand_dims(pi, 2)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi * vi).sum(axis=1)
        u = vi_attended + vq
        return u

class SANModel(nn.Cell):
    def __init__(self, emb_size=768, att_ff_size=512, output_size=17625,
                 num_att_layers=1):
        super(SANModel, self).__init__()
        self.image_channel = ImageEmbedding(output_size=emb_size)

        self.ques_channel = BertModel.load('bert-base-uncased')
        self.ques_channel.set_train(False)

        self.softmax = ops.Softmax()

        self.san = nn.CellList(
            [Attention(d=emb_size, k=att_ff_size)] * num_att_layers)

        self.mlp = nn.SequentialCell(
            nn.Dropout(keep_prob=0.5),
            nn.Dense(emb_size, output_size))

    def construct(self, questions, images):
        image_embeddings = self.image_channel(images)
        
        attention_mask = mnp.where(questions,1,0)
        attention_mask = Tensor(attention_mask,dtype=mindspore.int64)
        token_type_id = ops.zeros_like(questions)
        
        ques_embeddings = self.ques_channel(questions, attention_mask, token_type_id)[1]
        vi = image_embeddings
        u = ques_embeddings
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u)
        return self.softmax(output)