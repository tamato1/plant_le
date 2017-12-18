import os
import numpy as np 
import pandas as pd 
import keras
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model 
from keras.layers import Dense

#list_label = [f for f in os.listdir(direc) if not os.path.isfile(f)]

label_dict = ['Black-grass' ,'Charlock' ,'Cleavers' ,'Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent' ,'Maize' ,'Scentless Mayweed' ,'Shepherds Purse' ,'Small-flowered Cranesbill','Sugar beet']
img_size = 224




def read_file(direc):
	#add file to list
	list_label = label_dict
	filename = []
	label = []
	input_list =[]
	for lab in range(len(list_label)):
		for fil in os.listdir(direc+'/'+list_label[lab]):
			filename.append(direc+'/'+list_label[lab]+'/'+fil)
			label.append(lab)
			input_list.append([direc+'/'+list_label[lab]+'/'+fil,lab])
	return input_list

def read_images(input_list,batch_size = 50):
	while True:
		datas = []
		label = []
		for i in range(batch_size):
			file = np.random.choice(input_list)
			img = image.load_img( file[0] , target_size = (img_size,img_size))
			img = image.img_to_array(img)
			datas.append(img)
			label.append(file[1])
		datas = preprocess_input(datas)
		yield(np.array(datas) , np.array(label)

input_list = read_file('train/')



#define Model
base_model = VGG16(weights = 'imagenet',include_top = False)

md = base_model.output
md = Dense(1024, activation = 'relu')(md)
md = Dense(512, activation = 'relu')(md)
predict = Dense(12,activation = 'softmax')(md)

model = Model(inputs = base_model.input , outputs = predict)

for layer in base_model.layers:
	layer.trainable= False


#train parameter
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.fit_generator(read_images(input_list,batch_size = 20),steps_per_epoch = 1000 ,epochs =100)

