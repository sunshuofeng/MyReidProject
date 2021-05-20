Here I have provided several datasets for pedestrian re-recognition, and provided the code for processing the data. All datasets will be divided into three folders: 

* train
* query
* gallery


 **1. Market1501:** 

This is the blog introducing this dataset: [blog](https://blog.csdn.net/ctwy291314/article/details/83544088)

[Baidu Disk DownLoad:](https://link.csdn.net/?target=https%3A%2F%2Fpan.baidu.com%2Fs%2F1ntIi2Op)

**2.CUHK03:**

  This is the blog introducing this dataset: [blog](https://blog.csdn.net/ctwy291314/article/details/83544210)

  [DownLoad](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)

 **3.P-DESTRE**

 It is a pedestrian data set taken by drones, which can be used to solve the problem of pedestrian re-identification from the monitoring perspective. The data set provides tags such as pedestrian detection, pedestrian tracking, pedestrian re-identification, and pedestrian re-identification when changing clothes.

 [DownLoad:](http://p-destre.di.ubi.pt/)

 **！！ If you use the P-DESTRE dataset, in order to construct the structure of train, query, and gallery, you should**

``` python
python make_PData.py --root your_own_pdata_root
```

And the new root is data/Pdata

 **4.VC-Clothes**
 This data set contains a total of two different domains of data, one of which is a pedestrian modeled in the game for training. The other is a dataset of pedestrians changing clothes in the real world for eval

 [DownLoad](https://wanfb.github.io/dataset.html)