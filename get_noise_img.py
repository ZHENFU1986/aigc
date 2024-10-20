import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
from config import *
from PIL import Image
def save(img,path):
	img=img.clip(0,1)
	img=img*255
	img = Image.fromarray(img)
	img = img.convert('L')
	img.save(path)
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
x_train=x_train/255
result=[]
count=0
for x,y in zip(x_train,y_train):
    if y!=8:
        continue
    count+=1
    print (count)
    for t in range(1,T+1):
        rand=np.random.randn(x.shape[0],x.shape[1])
        a1=aerfa_m[t]**0.5
        a2=(1-aerfa_m[t])**0.5
        img=a1*x+a2*rand
        save(img,"noise_img/{}.jpg".format(t))
    break



