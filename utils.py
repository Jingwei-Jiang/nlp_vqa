from config import *
from PIL import Image
from mindspore import ops
import mindspore
import mindspore.dataset.vision.py_transforms as py_trans
from mindspore.dataset.transforms.py_transforms import Compose

def path_gen( train=False, val=False, test=False ):
    if train:
        split = 'train'
    elif val:
        split = 'val'
    else:
        split = 'test'
    ap = annotations_path + split + '_align.json'
    qp = questions_path + split +  '_align.json'
    ip = images_path + split + '/'
    return ap, qp, ip

def decode(image):
    return Image.fromarray(image)

def trans_gen( train=False, val=False, test=False ):
    mode = 'train' if train else 'val'
    # 定义transforms列表
    transforms_dict = {
        'train':[
            decode,
            py_trans.Resize(size=(224, 224)),
            py_trans.RandomHorizontalFlip(0.2),
            py_trans.ToTensor(),
            py_trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ],
        'val':[
            decode,
            py_trans.Resize(size=(224, 224)),
            py_trans.ToTensor(),
            py_trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]}

    # 通过Compose操作将transforms列表中函数作用于数据集图片
    return Compose(transforms_dict[mode])


def batch_accuracy(predicted, answers):
    """ Compute the accuracies for a batch of predictions and answers """
    print("predicted:", predicted)
    print("answers:", answers)
    arg_max = ops.Argmax(axis=1, output_type=mindspore.int32)
    gather = ops.GatherD()
    minimum = ops.Minimum()
    unsqueeze = ops.ExpandDims()
    squeeze = ops.Squeeze(1)

    predicted_index = arg_max(predicted)
    predicted_index = unsqueeze(predicted_index, 1)

    agreeing = gather(answers, 1, predicted_index)
    agreeing = squeeze(agreeing)
    return minimum(agreeing * 0.3, 1.0)