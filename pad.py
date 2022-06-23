from bert4ms.tokenizers.bert_tokenizer import *
import mindspore
import numpy as np
s = "here is some text to encode"

token = BertTokenizer.load('bert-base-uncased')

#inputs = mindspore.Tensor([token.encode(s, add_special_tokens=True)],mindspore.int32)
#len(inputs[0])
a = token.encode(s, add_special_tokens=True)
b = np.array(a)
c = np.pad(b,(0,128 - len(b)))
print(mindspore.Tensor([c]))