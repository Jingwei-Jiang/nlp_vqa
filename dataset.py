import json
import re

from PIL import Image
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
import numpy as np
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision.py_transforms as py_trans
from mindspore.dataset.transforms.py_transforms import Compose

import numpy as np
import mindspore.context as context

from utils import *
from config import *

_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

def process_punctuation(s):
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()

def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    questions = [[q['image_id'],q['question']] for q in questions_json['questions']]
    for question in questions:
        question[1] = question[1].lower()[:-1]
        yield [ question[0], question[1]]

def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))

class VQA:
    def __init__(self, train=False, val=False, test=False ):
        super(VQA, self).__init__()
        self.train = train
        self.val = val
        self.test = test
        
        self.answers_path, self.questions_path, self.imgs_path = path_gen( train, val, test )

        with open(self.questions_path, 'r') as fd:
            self.questions_json = json.load(fd)
        with open(self.answers_path, 'r') as fd:
            self.answers_json = json.load(fd)

        # q and a
        self.questions = list(prepare_questions(self.questions_json))
        self.answers = list(prepare_answers(self.answers_json))
        

    def img_path_gen(self, item):
        split = 'train' if self.train else 'val'
        img_path = self.imgs_path+'COCO_'+ split + '2014_' + str(item).zfill(12) + '.jpg'
        return img_path
    
    def __getitem__(self, item):
        q = self.questions[item]
        a = self.answers[item]
        path_img = self.img_path_gen(q[0])
        img = Image.open(path_img).convert('RGB')
        img = Tensor(np.array(img))
        return  q[1], a, img
        #return img
    
    def __len__(self):
        return len(self.questions)
    
def get_loader(train=False, val=False, test=False):

    split =VQA(train,val,test)
    loader = GeneratorDataset(
        split,
        column_names=["q","a","i"],
        shuffle=train
    )
    compose_trans = trans_gen(train, val, test)
    loader = loader.map(operations=compose_trans, input_columns="i")
    loader = loader.batch(batch_size=4,drop_remainder=True )
    loader.source = split
    return loader