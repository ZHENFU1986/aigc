#作者 卢菁 微信:13426461033
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
from config import *
def convert(img):
	result=[[round(s,3) for s in ss] for ss in img]
	return result

#tensorflow手写数字图像
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
#把像素值归一化成0~1
x_train=x_train/255
result=[]
count=0
for x,y in zip(x_train,y_train):
    if y!=8:
        continue
    count+=1
    print (count)
    #每张图片，生成100个训练样本
    for _ in range(0,100):
        #随机出噪声 e，这个是模型的预测目标
        rand=np.random.randn(x.shape[0],x.shape[1])
        t=random.randint(1,T)
        a1=aerfa_m[t]**0.5
        a2=(1-aerfa_m[t])**0.5
        #把x(t-1) 给计算出来，这也是模型的输入
        #原始图像和噪声加权求和，权重参考config文件
        input_img=a1*x+a2*rand
        result.append(str([convert(input_img),convert(rand),t]))
with open("train_data","w") as f:
	f.writelines("\n".join(result))





