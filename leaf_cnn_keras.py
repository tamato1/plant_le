import os
import numpy as np 
import pandas as pd 
import keras
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model 
from keras.layers import Dense,GlobalAveragePooling2D
from keras.utils import plot_model
import sklearn.preprocessing
from keras.callbacks import ModelCheckpoint,TensorBoard
#list_label = [f for f in os.listdir(direc) if not os.path.isfile(f)]

label_dict = ['Black-grass' ,'Charlock' ,'Cleavers' ,'Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent' ,'Maize' ,'Scentless Mayweed' ,'Shepherds Purse' ,'Small-flowered Cranesbill','Sugar beet']
img_size = 224
label_binary = sklearn.preprocessing.LabelBinarizer()
label_binary.fit(range(12))

def read_images(direc,batch_size = 50):
	print 'generator are working\n'
	while True:
		datas = []
		label = []
		for i in range(batch_size):
			folder = np.random.choice(len(label_dict))
			img_random = np.random.choice(os.listdir(direc+'/'+label_dict[folder]+'/'))
			img_name = direc+label_dict[folder]+'/'+img_random
			img = image.load_img( img_name , target_size = (img_size,img_size))
			img = image.img_to_array(img)

			datas.append(img)

			label.append(folder)
		label = label_binary.transform(label)
		datas = np.array(datas)
		datas = preprocess_input(datas)

		yield(datas , label)



#define Model
base_model = VGG16(weights = 'imagenet',include_top = False)

md = base_model.output
md = GlobalAveragePooling2D()(md)
md = Dense(1024, activation = 'relu')(md)
md = Dense(512, activation = 'relu')(md)
predict = Dense(12,activation = 'softmax')(md)

model = Model(inputs = base_model.input , outputs = predict)

for layer in base_model.layers:
	layer.trainable= False


#train parameter

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',metrics = ['accuracy'])
#print plot_model(model,show_shapes = True)
checkpoint = ModelCheckpoint(filepath ='/leaf_1/leaf_VGG16_1.hdf5',verbose = 1,save_best_only = True)
gen_input = read_images('train/', batch_size = 50)
model.fit_generator(generator = gen_input ,samples_per_epoch = 50*2 , steps_per_epoch = 100 ,epochs =200,callbacks = [checkpoint,])

