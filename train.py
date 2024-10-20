#作者 卢菁 微信:13426461033
import tensorflow
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Flatten,Dense,Activation,Input,Embedding,Reshape,Add,Conv2D,Concatenate,MaxPooling2D,Conv2DTranspose
import tensorflow.keras.backend as K
from config import *
def read_data(path):
	with open(path) as f:
		lines=f.readlines()
	lines=[eval(s.strip()) for s in lines]
	X1,Y,X2=zip(*lines)
	X1=np.array(X1)
	X2=np.array(X2)
	Y=np.array(Y)
	return X1,X2,Y

def get_m_t_img(t_img,channel_num,w):
    t_img_m=Conv2D(channel_num, (3, 3), activation='tanh',padding="same")(t_img)
    d=int(28/w)
    t_img_m=MaxPooling2D(pool_size=(d,d),strides=d,padding="valid")(t_img_m)
    return t_img_m

os.environ["CUDA_VISIBLE_DEVICES"]= "3"

X1,X2,Y=read_data("train_data")
#输入加噪声得到图片
input1=Input((28,28))
#输入一个时间信号 1~T
input2=Input(1,)

#1~T每个时间信号，都对应一个28*28*1的图像
t_img=Embedding(T+1, 28*28, input_length=1)(input2)
t_img=Reshape((28,28,1))(t_img)

img=Reshape((28,28,1))(input1)

#逐像素相加
#img具备了时间和图像的双重信息
img=Add()([t_img,img])
#一次标准的卷积核池化操作
#32通道，14*14
img=Conv2D(32, (3, 3), activation='tanh',padding="same")(img)
img=MaxPooling2D(pool_size=(2,2),strides=2,padding="valid")(img)

#让时间信号信息和图像的尺寸，channels相等,通过标准卷积池化操作
t_img_m=get_m_t_img(t_img,32,14)
#逐像素相加
img=Add()([t_img_m,img])
img=Conv2D(32, (3, 3), activation='tanh',padding="same")(img)
img=MaxPooling2D(pool_size=(2,2),strides=2,padding="valid")(img)

t_img_m=get_m_t_img(t_img,32,7)
img=Add()([t_img_m,img])

img= Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="tanh")(img)
t_img_m=get_m_t_img(t_img,32,14)
img=Add()([t_img_m,img])

img= Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="tanh")(img)
t_img_m=get_m_t_img(t_img,32,28)
img=Add()([t_img_m,img])

img=Conv2D(1, (1, 1),padding="same")(img)
img=Reshape((28,28))(img)

model=Model([input1,input2],img)
model.compile(loss="mse",optimizer="Adam",metrics=['mse'])
#X1 t时刻加噪声图像 X2 t时刻  Y预测噪声
model.fit([X1,X2],Y,batch_size=128,epochs=100)
model.save("model.h5")

