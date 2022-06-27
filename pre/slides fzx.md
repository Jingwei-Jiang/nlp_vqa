### 2.0 环境设置

1.同步文件

```
import moxing as mox
mox.file.copy_parallel(src_url="s3://focus/nlp/data/", dst_url='../data/')
mox.file.copy_parallel(src_url="s3://focus/nlp/nlp_vqa/", dst_url='.')
```

2.设置训练芯片和模式（context.GRAPH_MODE不支持初始化Tensor）

```python
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
```

### 2.2 img特征提取

直接输入图片（3x224x224）到搭建的卷积网络，输出特征（196x768），而非直接使用提取好的特征

```python
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
```

### 2.5 模型训练及验证

1.为了支持多个输入(`question`和`img`)，自定义`WithLossCell`

```python
class WithLossCell(nn.Cell):
    def __init__(self, model):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.loss = nn.SoftmaxCrossEntropyWithLogits()
        self.net = model

    def construct(self, q, a, img):
        out = self.net(q, img)
        loss = self.loss(out, a)
        return loss
```

2.定义训练网络

```python
#定义网络
model = san.SANModel()
#定义优化器
opt = nn.Adam(params=model.trainable_params())
#定义带Loss的网络
net_with_loss = WithLossCell(model)
#包装训练网络
train_net = TrainOneStepCell(net_with_loss, opt)
#设置训练模式
train_net.set_train(True)
```

3.定义验证网络，模型输出只要属于对应问题的十个答案之一即认为正确

```python
class WithAccuracy(nn.Cell):
    def __init__(self, model):
        super(WithAccuracy, self).__init__(auto_prefix=False)
        self.net = model

    def construct(self, q, a, img):
        out = self.net(q, img)
        out = ops.Argmax(output_type=mindspore.int32)(out)
        return out, a
#model即为训练网络中同一个model
eval_net = WithAccuracy(model)
#设置验证模式
eval_net.set_train(False)
```

4.每个batch计算准确率，最后求平均得到某个epoch的验证准确率

```python
out, a = eval_net(q, a, img)
predicted = out.asnumpy()
ans = a.asnumpy()
batch_size = ans.shape[0]
acc = 0
for i in range(batch_size):
	if ans[i,predicted[i]]!=0:
        acc += 1
accuracy = acc / batch_size
```

