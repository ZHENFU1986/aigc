import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random

from PIL import Image
def convert(img):
	result=[[round(s,3) for s in ss] for ss in img]
	return result

T=300
aerfa_m=[None]
aerfa=0.95
for i in range(1,T+1):
	aerfa_m.append(aerfa**i)

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
result=[]
count=0
for x,y in zip(x_train,y_train):
    if y!=8:
        continue
    count+=1
    print (count)
    img = Image.fromarray(x)
    img = img.convert('L')
    img.save("8/{}.jpg".format(count))





