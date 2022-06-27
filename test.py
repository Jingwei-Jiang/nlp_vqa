from dataset import *
import sys
import os.path
import mindspore
from mindspore import Tensor, nn, Model, context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.parameter import ParameterTuple
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
# from mindspore.nn import WithLossCell
import numpy as np
from tqdm import tqdm
import config
import dataset
import san
import utils
import mindspore.context as context
import json
import math
from datetime import datetime


loader=get_loader(train = True).create_dict_iterator()
# # print("Loader Created")

# # print("iter:0")
test = loader._get_next()
print(test['q'].shape)
print(test['a'])
print(test['img'].shape)

# # print("iter:1")
# test = loader._get_next()
# print(test['q'].shape)
# # print(test['a'])
# # print(test['img'].shape)
# q_array = test['q'].asnumpy()
# print(q_array[0])