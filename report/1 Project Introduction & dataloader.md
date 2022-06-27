## 1 Project Introduction

### 1.1 工作简介

VQA：以一张图片和一个关于图片内容的自然语言形式的问题作为输入，要求输出正确答案

数据集： `VQAv2`

小结：属于一种多标签分类的问题，计算损失的时候采用多标签损失。



### 1.2 开发环境

+ 开发工具：`ModelArts Ascend Notebook`环境，选用`Ascend910`芯片作为训练芯片

+ 开发包、开源库：

  1. Mindspore1.3.0
  2. numpy

+ 系统运行要求：

  `python3.7.5`与可运行`Mindspore1.3.0`的开发环境



## 2 DataLoader

### 2.1 数据集简介

VQAv2数据集的构成如下：

- 1张图片有大概5个问题
- 1个问题有10个答案
- test没有annotation文件

#### Question

```python
question{
"question_id" : int,  #问题id
"image_id" : int,     #问题对应的图片id
"question" : str      #具体的问题
}
```

#### Annotation

具有数据结如下：

```python
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

#### 2.2 数据预处理

+ 问题与答案对齐

+ 图片与问题对齐

#### 2.3 数据集加载

数据类型与预处理如下：

1. `img`：图片为三通道`RGB`模式，加载成三维的`Tensor`即可
2. `question`：预处理，进行词形还原，大小写转换等，再通过预训练的`Tokenizer`进行`one-hot`编码，扩充成定长向量输出。
3. `Answers`：预处理，进行词形还原，大小写转换等，自己构造词汇表进行`one-hot`编码

处理结果：

![dataloader](D:\Grade3\大三下\nlp\pro\report\dataloader.png)