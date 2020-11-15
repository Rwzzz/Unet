# Unet


@[TOC](文章目录)


<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 前言

<font color=#999AAA >Unet多应用于CNN中图像分割领域。对于小数据集也有很好的性能。



# 一、Unet
##  1.Unet网络框架
&ensp;&ensp;&ensp;论文中只用分割出细胞边界，所以最后使用的是2个1\*1卷积得到背景和目标两个。如果是多目标分割，根据分割目标的种类来决定使用1*1的卷积的数量来输出Segmentation map.<font color=red>注：对于多目标的Label标注：可以使用不同颜色，然后使用One-hot编码生成Label进行训练。<font>
![Unet network architecture](https://img-blog.csdnimg.cn/20201114113549401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMTM2NTQy,size_16,color_FFFFFF,t_70#pic_center)
## 2.Unet运用的 Skip connection
语义分割网络在特征融合时也有2种办法：
（1）FCN式的逐点相加，也叫加操作。如图1。
（2）U-Net式的channel维度拼接融合，也叫叠操作。如图2。

![FCN的Skip connection](https://img-blog.csdnimg.cn/20201114113906740.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMTM2NTQy,size_16,color_FFFFFF,t_70#pic_center)
  
  <center> 图1 FCN中Skip connection方式 </center>
  
![Unet的Skip connection](https://img-blog.csdnimg.cn/20201114113734874.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMTM2NTQy,size_16,color_FFFFFF,t_70#pic_center)
  <center> 图2 Unet中Skip connection方式 </center>

<font color=red> Copy and crop操作：</font> 经过两次3\*3卷积后，大小变为28×28，然后经过一次2\*2卷积，变为56×56，这时和左侧大小为64×64的图像进行维度的叠加，但是由于图像大小不同，需要将左侧的64×64大小的图像裁剪为56×56大小,此处的裁剪是合理的，原因看下一步3中讲的Overlap-tile策略。
## 3.Unet应用的Overlap-tile策略
&ensp;&ensp;&ensp;可以发现Unet论文中输入的图像是572×572但是输出图像确实388×388.这是不是就意味着原图像存在信息丢失的现象呢？实际上不是的，经过了no padding的卷积操作，输入图像和输出图像肯定是不一样的尺寸，但是Unet在论文中提及了一种策略--Overlap-tile，将图像进行镜像扩充和输入网络，这样经过卷积后得到的输出图像和实际需要提取的图像是相同的尺寸。
&ensp;&ensp;&ensp;<font color = red>例如下图，实际需要分割的图像是黄色框所选中部分，但是输入到网络中的图像是蓝色部分，对空白部分进行镜像填充，这样经过网络后所得到的的输出大小尺寸适合实际需要分割的图像大小是一样的。<front>
![Overkap-tile](https://img-blog.csdnimg.cn/20201115132210215.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMTM2NTQy,size_16,color_FFFFFF,t_70#pic_center)
## 4.Unet的LossFunction
<font color=red>**&ensp;&ensp;&ensp;Unet训练时对细胞边界的像素点增加了权重，使用加权损失函数，可以更注重细胞边界分割。此处d1,d2个人不清楚是使用什么计算的距离**<font>
![LossFunction](https://img-blog.csdnimg.cn/20201115133731278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMTM2NTQy,size_16,color_FFFFFF,t_70#pic_center)

# 二、Pytorch环境搭建及Training
## 1.相关资源
数据集：[https://github.com/Rwzzz/Unet](https://github.com/Rwzzz/Unet)
代码：[https://github.com/Rwzzz/Unet](https://github.com/Rwzzz/Unet)

## 2.实验结果

 - 训练集大小30张图片
 - 训练时间epochs=40,batch_size=1
 - 训练环境 pytorch1.7 
 
在实际训练中为了方便，没有采用Unet中的策略。
统一输入和输出尺寸的两种方案：
1.padding='same'形式
2.对小分辨率特征图进行填充后进行维度的连接。

![Loss值](https://img-blog.csdnimg.cn/20201115142923672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMTM2NTQy,size_16,color_FFFFFF,t_70#pic_center)

<center > 图1.训练的Loss</center>


![分割图](https://img-blog.csdnimg.cn/20201115143327375.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMTM2NTQy,size_16,color_FFFFFF,t_70#pic_center)

<center > 图2.测试集预测结果</center>

# 总结
&ensp;&ensp;&ensp;&ensp;Unet模型简单，并且使用较少的数据集，可以达到非常理想的分割效果，对于医学和其他一些数据集比较少的领域优势很大。
