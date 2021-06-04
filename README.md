   This is the first code library I wrote myself, the purpose is to improve my ability in reid. 
    Since it is the first time to write, the code style may not be very good, but try to make the code more readable and extensible, hope to help you

# purpose
   Regarding this library, my original intention is not to build a project like fastreid, Torchreid, which is highly scalable, practical, and widely used.         After all, I am only a student, and I have not yet possessed such capabilities. The purpose of establishing this library is more to exercise one's own ability in this area by reproducing various papers and modifying other people's codes. So in terms of code, the scalability may not be strong, and there may be many mistakes. I hope everyone who reads the code can help point out. If you have a better implementation method or some better tricks, you can put them forward and discuss them together. My qq is 2276153406 and my email is 2276153406@qq.com
   
 关于这个库，我的初衷不是建立像fastreid，Torchreid这样的项目，该项目具有高度的可扩展性，实用性和广泛的用途。毕竟，我只是一个大三学生，而我还没有具备这样的能力。建立该库的目的更多是通过复现各种论文，修改他人的代码来提高自己在该领域的能力。因此，就代码而言，可扩展性可能不强，并且可能存在许多错误。希望所有阅读该代码的人都能帮助指出。如果您有更好的实现方法或一些更好的技巧，则可以提出它们并进行讨论。我的qq是2276153406，我的电子邮件是2276153406@qq.com
# Requirements

``` python
pip install git+https://github.com/pabloppp/pytorch-tools -U
pip install nncf
pip install ray[tune]
pip install fastai
pip install thop
pip install efficientnet_pytorch

```

**!!!!!目前代码库开始重构以及测试，完整的训练管道还未能使用，但是可以根据需要取所需的代码，重构完后会在知乎上通知，谢谢大家**

**==想知道我的github做了什么工作，请到我的知乎专栏去查看目前已经完成的工作 #F44336==：**

[我的知乎专栏](https://zhuanlan.zhihu.com/p/373137077)


# About this repo:
[1.BaseLine](#Baseline)

[2.Project](#Project)


# Baseline
   First Baseline contains all the parts needed to build a Reid model, such as the reading of multiple Reid datasets, multiple common losses, multiple optimizations and lr_scheduler, and the calculation of MAP. 
   Then the model is Strong Baseline. 


首先baseline包含了构建一个reid模型所需要的所有部分，比如多种reid数据集的读取，多种常用loss，多种优化器和学习率调整,以及map的计算，然后baseline的模型是strong baseline。这部分最主要是为了小白入门使用的，以及用于project实现论文代码的。



	
	
# Project
The project contains the code that I reproduced when I read the paper, and is usually implemented without an official code. Of course, for some of the papers that provide the code, I may migrate here with some minor changes.

project包含了我阅读论文时复现的代码，通常都是在没有官方代码的情况下实现。当然，对于有些提供了代码的论文，我可能会稍做修改迁移到这里。

详情可以看我的知乎专栏，对于复现的论文都会有相应的博客进行解析


**1.Clothe Change Problem**

[Unofficial implement of the papaer :
"Person Re-identification by Contour Sketch
under Moderate Clothing Change"](./projects/Sketch)

**2.Clothe Change Problem**

[Unofficial implement of the Paper: 
"Learning Shape Representations for
Person Re-Identification under Clothing Change "](./projects/Shape)

**3.**

[Unofficial implement of the Paper:"Receptive Multi-granularity Representation for
Person Re-Identification"](./projects/RMGL)

**4.**

[Unofficial implement of the Paper:"Style Normalization and Restitution for Generalizable Person Re-identification"](./projects/SNR)

