This is the unofficial implement of the paper:[Person Re-identification by Contour Sketch
under Moderate Clothing Change](https://arxiv.org/abs/2002.02295)

![Model](https://github.com/sunshuofeng/MyReidProject/blob/main/images/Sketch0.PNG)


It should be noted that only the PRCC dataset can be used to train this network:[download](https://www.isee-ai.cn/~yangqize/clothing.html)


**如果代码出错，请到下面的kaggle notebook 中进行实验，这是我复现代码时正确运行的代码**
[code](https://www.kaggle.com/houssad/notebooka016ed1dbc)



# Training on PRCC
First you should 

``` python
python make_data.py --root prcc
```
root is the PRCC dataset folder

And my code will divide PRCC dataset into three folder:
1.train
2.query
3.gallery

And three folders are in 'datasets/prcc-proprecess'

After that,you can train the model


For common training:

``` python
python main.py 
```


