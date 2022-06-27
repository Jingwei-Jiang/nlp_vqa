import json
import re
import json
import utils
import itertools
from collections import Counter

from PIL import Image
import mindspore
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
from bert4ms.tokenizers.bert_tokenizer import *

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
    # print("ok")
    questions = [[q['image_id'], q['question']] for q in questions_json['questions']]
    for question in questions:
        question[1] = question[1].lower()[:-1]
        yield [question[0], question[1]]


def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))


def ans_vocab_gen():
    print("Generating answers vocab...")
    ans_path, ques_path, image_path = path_gen(train=True)

    with open(ans_path, 'r') as fd:
        answers = json.load(fd)

    answer_lists = prepare_answers(answers)

    all_tokens = itertools.chain.from_iterable(answer_lists)
    counter = Counter(all_tokens)
    ans = counter.keys()
    # 先按个数多少排序，再按字典序排序
    tokens = sorted(ans, key=lambda x: (counter[x], x), reverse=True)
    ans_to_idx = {t: i for i, t in enumerate(tokens)}
    idx_to_ans = {i: t for i, t in enumerate(tokens)}
    print("Answers vocab is generated")
    return ans_to_idx, idx_to_ans


class VQA_dataset:
    def __init__(self, train=False, val=False, test=False):
        super(VQA_dataset, self).__init__()
        self.train = train
        self.val = val
        self.test = test

        self.answers_path, self.questions_path, self.imgs_path = path_gen(train, val, test)

        with open(self.questions_path, 'r') as fd:
            self.questions_json = json.load(fd)
        with open(self.answers_path, 'r') as fd:
            self.answers_json = json.load(fd)

        self.questions = list(prepare_questions(self.questions_json))
        self.answers = list(prepare_answers(self.answers_json))
        self.tokenizer = BertTokenizer.load('bert-base-uncased')
        self.ans_to_idx, _ = ans_vocab_gen()
        self.ans_vocab_len = len(self.ans_to_idx)

        for i, item in enumerate(self.answers):
            ans_encoding = np.zeros(self.ans_vocab_len).astype(np.float32)
            for ans in item:
                if ans in self.ans_to_idx.keys():
                    ans_encoding[self.ans_to_idx[ans]] += 1.0
            self.answers[i] = ans_encoding / sum(ans_encoding)

    def img_path_gen(self, item):
        split = 'train' if self.train else 'val'
        img_path = self.imgs_path + 'COCO_' + split + '2014_' + str(item).zfill(12) + '.jpg'
        return img_path

    def __getitem__(self, idx):
        q = self.questions[idx]
        a = self.answers[idx]
        path_img = self.img_path_gen(q[0])
        img = Image.open(path_img).convert('RGB')
        img = np.array(img)
        question_token = self.tokenizer.encode(q[1], add_special_tokens=True)
        token_array = np.array(question_token)
        token_array = np.pad(token_array, (0, 128 - len(token_array)))

        return token_array, a, np.array(img)

    def __len__(self):
        return len(self.questions)


def get_loader(train=False, val=False, test=False):
    vqa_dataset = VQA_dataset(train, val, test)
    loader = GeneratorDataset(
        vqa_dataset,
        column_names=["q", "a", "img"],
        shuffle=train
    )
    compose_trans = trans_gen(train, val, test)
    loader = loader.map(operations=compose_trans, input_columns="img")
    # 去除不足一个batch的数据
    loader = loader.batch(batch_size=batch_size, drop_remainder=True)
    loader.source = vqa_dataset
    return loader
