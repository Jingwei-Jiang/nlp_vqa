import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from bert4ms.models import BertModel

def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same')


def _conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same')


def _conv7x7(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same')


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class BasicBlock(nn.Cell):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.SequentialCell(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel)])
        self.add = ops.TensorAdd()

    def forward(self, x):
        identity = x

        out = self.left(x)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
         ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel)])
        self.add = ops.TensorAdd()

    def construct(self, x): # pylint: disable=missing-docstring
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.

    Examples:
         ResNet(ResidualBlock,
                [3, 4, 6, 3],
                [64, 256, 512, 1024],
                [256, 512, 1024, 2048],
                [1, 2, 2, 2],
                10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.

        Returns:
            SequentialCell, the output layer.

        Examples:
             _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x): # pylint: disable=missing-docstring
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


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
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.channels * 2, self.channels*4, kernel_size=3, stride=1, padding=0, pad_mode='same'),
            nn.BatchNorm2d(self.channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.channels*4, output_size, kernel_size=3, stride=1, padding=0, pad_mode='same')
        ])

        #resnet18 7x7x512
        # output_size = 500
        # self.resnet18 = ResNet(BasicBlock,
        #                        [2, 2, 2, 2],
        #                        [64, 64, 128, 256],
        #                        [64, 128, 256, 512],
        #                        [1, 2, 2, 2],
        #                        output_size)
        # self.resnet34 = ResNet(BasicBlock,
        #                        [3, 4, 6, 3],
        #                        [64, 64, 128, 256],
        #                        [64, 128, 256, 512],
        #                        [1, 2, 2, 2],
        #                        output_size)
        # resnet50 7x7x2048
        # self.resnet50 = ResNet(ResidualBlock,
        #                        [3, 4, 6, 3],
        #                        [64, 256, 512, 1024],
        #                        [256, 512, 1024, 2048],
        #                        [1, 2, 2, 2],
        #                        output_size)
        # resnet101 7x7x2048
        # self.resnet50 = ResNet(ResidualBlock,
        #                        [3, 4, 23, 3],
        #                        [64, 256, 512, 1024],
        #                        [256, 512, 1024, 2048],
        #                        [1, 2, 2, 2],
        #                        output_size)
        # resnet152 7x7x2048
        # self.resnet50 = ResNet(ResidualBlock,
        #                        [3, 8, 36, 3],
        #                        [64, 256, 512, 1024],
        #                        [256, 512, 1024, 2048],
        #                        [1, 2, 2, 2],
        #                        output_size)
    def construct(self, x):
        # return self.resnet50(x)
        # return self.resnet18(x)
        x = self.simple_cnn(x)
        N = x.shape[0]
        # transpose = ops.Transpose()
        return x.reshape((N, 196, self.output_size))

# class QuesEmbedding(nn.Cell):
#     def __init__(self, input_size=500, output_size=1024, num_layers=1, batch_first=True):
#         super(QuesEmbedding, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size,
#                             hidden_size=output_size, batch_first=batch_first)
#
#     def forward(self, ques):
#         # seq_len * N * 500 -> (1 * N * 1024, 1 * N * 1024)
#         _, hx = self.lstm(ques)
#         # (1 * N * 1024, 1 * N * 1024) -> 1 * N * 1024
#         h, _ = hx
#         ques_embedding = h[0]
#         return ques_embedding

class Attention(nn.Cell):
    def __init__(self, d=768, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Dense(d, k)
        self.ff_ques = nn.Dense(d, k)
        if dropout:
            self.dropout = nn.Dropout(keep_prob=0.5)
        self.ff_attention = nn.Dense(k, 1)
        self.expand_dims = P.ExpandDims()

    def forward(self, vi, vq):
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
        pi = nn.Softmax(ha)
        pi = self.expand_dims(pi, 2)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi * vi).sum(axis=1)
        u = vi_attended + vq
        return u

class SANModel(nn.Cell):
    def __init__(self, ques_vocab_size = 10000, word_emb_size=500, emb_size=768, att_ff_size=512, output_size=17625,
                 num_att_layers=1):
        super(SANModel, self).__init__()
        self.image_channel = ImageEmbedding(output_size=emb_size)

        # self.word_emb_size = word_emb_size
        # self.word_embeddings = nn.Embedding(ques_vocab_size, word_emb_size)
        # self.ques_channel = QuesEmbedding(
        #     word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)
        self.ques_channel = BertModel.load('bert-base-uncased')
        self.ques_channel.set_train(False)

        self.san = nn.CellList(
            [Attention(d=emb_size, k=att_ff_size)] * num_att_layers)

        self.mlp = nn.SequentialCell(
            nn.Dropout(keep_prob=0.5),
            nn.Dense(emb_size, output_size))

    def forward(self, questions, images):
        image_embeddings = self.image_channel(images)
        # embeds = self.word_embeddings(questions)
        # nbatch = embeds.size()[0]
        # nwords = embeds.size()[1]

        ques_embeddings = self.ques_channel(questions)[1]
        vi = image_embeddings
        u = ques_embeddings
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u)
        return output