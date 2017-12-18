import os
import numpy as np 
import pandas as pd 
import keras
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

direc = 'train/'
list_label = [f for f in os.listdir(direc) if not os.path.isfile(f)]

label_dict = ['Black-grass' ,'Charlock' ,'Cleavers' ,'Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent' ,'Maize' ,'Scentless Mayweed' ,'Shepherds Purse' ,'Small-flowered Cranesbill','Sugar beet']



list_label = label_dict
filename = []
label = []

#add file to list
for lab in range(len(list_label)):
    for fil in os.listdir(direc+'/'+list_label[lab]):
        filename.append(direc+'/'+list_label[lab]+'/'+fil)
        label.append(lab)
def read_images(input_list):
	datas = []
	input_list = sess.run(input_list)
	
	for file in input_list:
		img = image.load_img( file , target_size = (img_size,img_size))
		img = image.img_to_array(img)
		datas.append(img)
	return np.array(datas)