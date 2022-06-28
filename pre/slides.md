---
# try also 'default' to start simple
theme: academic
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: false
# some information about the slides, markdown enabled
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# persist drawings in exports and build
drawings:
  persist: false
---

# Visual Question Answering: SAN

Presentation slides for DL4NLP project

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

---

# Project Introduction

## 工作简介

- **VQA** - 以一张图片和一个关于图片内容的自然语言形式的问题作为输入，要求输出正确答案
- **Dataset** - VQAv2
- **Summary** - 属于一种多标签分类的问题，计算损失的时候采用多标签损失

## 开发环境

- 开发工具：ModelArts Ascend Notebook环境，选用Ascend910芯片作为训练芯片
- 开发包、资源库
  - Mindspore1.3.0
  - numpy
- 系统运行要求: python3.7.5 与可运行 Mindspore1.3.0 的开发环境

<br>
<br>

Read more about [Our Repository](https://github.com/frank-k666/nlp_vqa)

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: two-cols
---

::default::

# DataLoader

## 数据集简介

\
\
\
VQAv2数据集的构成如下：
- 1张图片有大概5个问题
- 1个问题有10个答案
- test没有annotation文件


::right::

### Question

```
question{
"question_id" : int,  #问题id
"image_id" : int,     #问题对应的图片id
"question" : str      #具体的问题
}
```

### Annotations

```
annotation{
"question_id" : int,
"image_id" : int,
"question_type" : str,          #问题类型
"answer_type" : str,			#答案类型
"answers" : [answer],
"multiple_choice_answer" : str
}
------------------------------
answer{
"answer_id" : int,
"answer" : str,				   #具体答案
"answer_confidence": str
}
```

---

# DataLoader

## 数据预处理

- 问题与答案对齐
- 图片与问题对齐


## 数据集加载
数据类型与预处理如下：

1. img: 图片为三通道RGB模式，加载成三维的Tensor即可
2. question：预处理，进行词形还原，大小写转换等，再通过预训练的Tokenizer进行one-hot编码，扩充成定长向量输出。
3. Answers：预处理，进行词形还原，大小写转换等，自己构造词汇表进行one-hot编码

<img src="/dataloader.png" style="width:400px;height:200px;position:absolute; left: 500px; top: 75px"/>

---
layout: two-cols
---

::default::

# Text Embedding: LSTM

According to the paper we refer, the text embdding part is LSTM, which is easy to implemented.

```python
from mindspore.nn import LSTM
lstm = LSTM(input_size, hidden_size, num_layers)
```

::right::
\
\
\
\
LSTM Shortcome:

* Good RNN variant, but still not good at handling long sequence.
* Only "look forward", not able to "look back"

\
\
Thus, we consider to use BERT


<img src="/LSTM model.png" style="width:450px;height:150px;position:absolute; left: 30px; top: 275px"/>



<style>
.footnotes-sep {
  @apply mt-20 opacity-10;
}
.footnotes {
  @apply text-sm opacity-75;
}
.footnote-backref {
  display: none;
}
</style>

---

# Text Embedding: BERT

```python {all|3-7|8-11|12-18|all}
class BertModel(BertPretrainCell):
    def construct:
        # Embedding
        token_embedding, segment_embedding, position_embedding = \
            BertEmbeddings(input_ids)
        embedding_output = token_embedding + 
            segment_embedding + position_embedding
        # BertEncoder == Transformer
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask)
        # sequence_output (batch_size, len(sequence), embedding_size)
        # Embedding of every word
        sequence_output = encoder_output[0]
        # pooled_output (batch_size, embedding_size)
        # Embedding of the input sequence
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


```

<img src="/bert.png" style="width:200px;height:200px;position:absolute; left: 670px; top: 75px"/>

<img src="/bert emb.png" style="width:400px;height:130px;position:absolute; left: 570px; top: 55%"/>

<style>
.footnotes-sep {
  @apply mt-20 opacity-10;
}
.footnotes {
  @apply text-sm opacity-75;
}
.footnote-backref {
  display: none;
}
</style>

---
layout: center
class: text-center
---

# Thanks
