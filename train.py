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
from mindspore.nn.loss.loss import _Loss
from mindspore.nn import WithLossCell
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

# import moxing as mox
# mox.file.copy_parallel(src_url="s3://focus/nlp/data/", dst_url='../data/')
# mox.file.copy_parallel(src_url="s3://focus/nlp/nlp_vqa/", dst_url='.')

class NLLLoss(_Loss):
    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__(reduction)
        self.reduce_sum = ops.ReduceSum()
        self.log_softmax = ops.LogSoftmax(axis=0)

    def construct(self, logits, label):
        nll = -self.log_softmax(logits)
        loss = self.reduce_sum(nll * label / config.alter_ans_num, axis=1).mean()
        return self.get_loss(loss)


class WithLossCell(nn.Cell):
    """
    The cell wrapped with NLL loss, for train only
    """
    def __init__(self, model):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._loss_fn = NLLLoss()
        self.net = model

    def construct(self, q, a, img):
        out = self.net(q, img)
        loss = self._loss_fn(out, a)
        return loss


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True)
        self.sens = sens

    def construct(self, q, a, img):
        weights = self.weights
        loss = self.network(q, a)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(q, a, img, sens)
        return F.depend(loss, self.optimizer(grads))


class TrainNetWrapper(nn.Cell):
    """
    The highest level train cell. (use it directly)
    """

    def __init__(self, model):
        super(TrainNetWrapper, self).__init__(auto_prefix=False)
        self.net = model

        self.loss_net = WithLossCell(self.net)
        optimizer = nn.Adam(params=self.net.trainable_params(), learning_rate=config.initial_lr)

        self.loss_train_net = TrainOneStepCell(self.loss_net, optimizer)

    def construct(self, q, a, img):
        loss = self.loss_train_net(q, img)
        out = self.net(q, a)
        accuracy = utils.batch_accuracy(out, a)
        return loss, accuracy


class OutLossAccuracyWrapper(nn.Cell):
    """
    The highest level cell for evaluation, wrapped with NLL Loss and accuracy. (use it directly)
    
    Output:
        output: a Tensor of shape (batch_size, config.max_answers) (logits)
        loss: a scalar value
        accuracy: a Tensor of shape (batch_size, 1)
    """
    def __init__(self, model):
        super(OutLossAccuracyWrapper, self).__init__()
        self.net = model
        self._loss_fn = NLLLoss()

    def construct(self, q, a, img):
        output = self.net(q, img)
        loss = self._loss_fn(output, a)
        accuracy = utils.batch_accuracy(output, a)
        return output, loss, accuracy


def run(net, loader, epoch, train=False, prefix=''):
    """ Run an epoch over the given loader """
    arg_max = ops.Argmax(axis=1, output_type=mindspore.int32)
    cat = ops.Concat(axis=0)
    losses = []
    accs = []

    if train:
        net.set_train(True)
    else:
        net.set_train(False)
        answers = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0, total=math.ceil(len(loader.source) / config.batch_size))
    for q, a, img in tq:
        if train:
            loss, acc = net(q, a, img)
        else:
            output, loss, acc = net(q, a, img)
            answer = arg_max(output)
            answers.append(answer.view(-1))
        losses.append(loss.view(-1))
        accs.append(acc.view(-1))
    answers = list(map(int, list(cat(answers).asnumpy())))
    accs = list(cat(accs).asnumpy().astype(float))
    if not train:
        return answers, accs
    else:
        return losses, accs


if __name__ == '__main__':
    # if config.device == 'GPU': os.environ['CUDA_VISIBLE_DEVICES'] = '1' # select GPU if necessary
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target='Ascend')

    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.ckpt'.format(name))
    print('The model will be saved to {}'.format(target_name))

    val_loader = dataset.get_loader(val=True)

    model = san.SANModel()

    # if config.pretrained:
    #     pretrain_params = load_checkpoint(config.pretrained_model_path)
    #     if pretrain_params is not None:
    #         print("Successfully loaded pretrained model from {}.".format(config.pretrained_model_path))
    #     load_param_into_net(SAN, pretrain_params)

    train_net = TrainNetWrapper(model) # for train
    eval_net = OutLossAccuracyWrapper(model) # for evaluation
    step = 0

    for epoch in range(config.epochs):
        train_loader = dataset.get_loader(train=True)
        
        """
        Wrapped train with `tqdm`
        """
        run(train_net, train_loader,train=True, prefix='train', epoch=epoch)
        answers, accs = run(eval_net, val_loader, train=False, prefix='val', epoch=epoch)
        
        # Calculate the validate accuracy mean of each batch
        total_acc = 0
        for acc_list in accs:
            total_acc += sum(acc_list)
        total_acc /= len(accs)*len(accs[0])

        results = {
            'name': name,
            # 'tracker': tracker.to_dict(),
            'accuracy': total_acc,
            'eval': {
                'answers': answers,
                'accuracies': accs
            },
            'vocab': train_loader.source.ans_to_idx,
        }

        # Save model as CKPT every 5 epochs
        if epoch % 5 == 0:
            mindspore.save_checkpoint(train_net.net, ckpt_file_name=os.path.join('logs', '{}.ckpt'.format(name)))
        
        # Save train meta info as JSON
        with open(os.path.join('logs', 'TrainRecord_{}.json'.format(name)), 'w') as fp:
            fp.write(json.dumps(results))

