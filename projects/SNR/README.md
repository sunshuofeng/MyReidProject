**想要训练SNR非常简单，只需要将这个model.py替换到baseline的models/model.py  然后直接按照baseline的训练方法  **

``` python
python main.py
```
**当然需要注意的是，要记住baseline默认的optimizer与论文不一样，要自己设置好参数**

**同时，baseline默认在market1501上训练，在market1501上测试，而SNR是跨域模型，要注意修改训练的数据集和测试的数据集**