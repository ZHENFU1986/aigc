#作者 卢菁 微信:13426461033
import tensorflow
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import *
from PIL import Image
from config import *
def save(img,path):
	img=img.clip(0,1)
	img=img*255
	img = Image.fromarray(img)
	img = img.convert('L')
	img.save(path)

model=load_model("model.h5")
#第T时刻的噪声图像
img=np.random.randn(28,28)
for i in range(T,0,-1):
        a1=aerfa_m[i]**0.5
        a2=(1-aerfa_m[i])**0.5
        #逐步预测噪声
        e=model.predict([np.array([img]),np.array([i])])[0]
        e2=(1-aerfa[i])*e/a2
        #通过公式逐步复原上一时刻
        img=(img-e2)/(aerfa[i]**0.5)
        save(img,"result/{}.jpg".format(i))
