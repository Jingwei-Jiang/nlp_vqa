# nlp_vqa
### 环境说明
mindspore支持cpu运行
若要在cpu运行，建议在python3.8环境下安装mindspore1.7.0

### 文件说明
+ `data.align.py`：数据预处理，QA按图片id对齐
+ `data.match.py`：数据预处理，删除没有对应图片的问题
+ `dataset`：数据集加载
+ `config`：超参数配置

### 运行说明
建议将数据集放在与本文件同一目录下，不然需要将`config`的`path_data`更改为对应路径
