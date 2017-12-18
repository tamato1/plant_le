

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd 
import os 
import cv2
import bottleneck as bn
from keras.models import Sequential
from scipy.misc import imread
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.layers import Dense, Activation
from scipy.misc import imresize
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)
img_size = 224
batch_size = 20
label_dict1 = {
	'Black-grass' : 0 ,
	'Charlock' : 1,
	'Cleavers' : 2,
	'Common Chickweed' : 3,
	'Common wheat':4,
	'Fat Hen': 5,
	'Loose Silky-bent' : 6,
	'Maize' :7,
	'Scentless Mayweed' : 8,
	'Shepherds Purse' : 9,
	'Small-flowered Cranesbill' : 10,
	'Sugar beet' :11 
	}

label_dict = ['Black-grass' ,'Charlock' ,'Cleavers' ,'Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent' ,'Maize' ,'Scentless Mayweed' ,'Shepherds Purse' ,'Small-flowered Cranesbill','Sugar beet']

def pretrained_feature(input_tensor):
	model = VGG16(weights='imagenet', include_top=False)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		input_img = sess.run(input_tensor)

		coord.request_stop()
		coord.join(threads)

		train_img=preprocess_input(input_img)
		features_train=model.predict(train_img)
		return features_train


def read_labeled_image_list(direc):
	#get folder name
	#list_label = [f for f in os.listdir(direc) if not os.path.isfile(f)]
	list_label = label_dict
	filename = []
	label = []

	#add file to list
	for lab in range(len(list_label)):
		for fil in os.listdir(direc+'/'+list_label[lab]):
			filename.append(direc+'/'+list_label[lab]+'/'+fil)
			label.append(lab)
	return np.array(filename), np.array(label)

def read_images(input_list):
	datas = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		input_list = sess.run(input_list)
		coord.request_stop()
		coord.join(threads)
		for file in input_list:
			img = image.load_img( file , target_size = (img_size,img_size))
			img = image.img_to_array(img)
			datas.append(img)
		return np.array(datas)

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	
	input_array = tf.convert_to_tensor(read_images(features["x"]), dtype = tf.float32)
	input_tensor = tf.reshape(input_array, [-1,img_size,img_size,3])
	pretrain = pretrained_feature(input_tensor)
	# Dense Layer
	pool_flat = tf.reshape(pretrain, [-1, 7*7*512])
	dense1 = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
	dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)
	dense3 = tf.layers.dense(inputs=dense2, units=128, activation=tf.nn.relu)
	dropout_last = tf.layers.dropout(inputs=dense3, rate=0.2 , training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout_last, units=2)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
	
	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
		train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	# Load training and eval data
	image_list, label_list = read_labeled_image_list('train/')
	#label_list = np.eye(np.max(label_list)+1)[label_list].astype(np.int32)
	# Create the Estimator
	x_train, x_test, y_train, y_test = train_test_split(image_list,label_list,test_size = 0.5, random_state = 40)
	diabetic_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="leaf1")

	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	train_input_fn =  tf.estimator.inputs.numpy_input_fn(x={"x": x_train},y=y_train,batch_size= batch_size,num_epochs=500,shuffle=True)
	diabetic_classifier.train(input_fn=train_input_fn,steps=800000,hooks=[logging_hook])
	# Evaluate the model and print results
	#test_list,test_labels = read_labeled_image_list('test.csv')
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_test},y=y_test,num_epochs=1,shuffle=False)
	eval_results = diabetic_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()
