import tensorflow as tf 
import numpy as np 
import model as M 
import pandas as pd 
import matplotlib.pyplot as plt 

def read_data():
	file = pd.read_csv('data.csv')
	x = file['x']
	y = file['y']
	label = file['class']
	coord = list(zip(x,y))
	return coord,label

def build_model(input_data):
	mod = M.Model(input_data,[None,2])
	mod.fcLayer(1)
	return mod.get_current_layer()

def build_graph():
	input_placeholder = tf.placeholder(tf.float32,[None,2])
	label_placeholder = tf.placeholder(tf.float32,[None,1])
	
	output = build_model(input_placeholder)

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_placeholder,logits=output))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	correct_pred = tf.equal(tf.round(tf.sigmoid(output)),tf.round(label_placeholder))
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

	return input_placeholder, label_placeholder, output, loss, train_step, accuracy

MAX_ITER = 10000
input_placeholder, label_placeholder, output, loss, train_step, accuracy = build_graph()
x,y = read_data()
y = [[i] for i in y]

with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	for i in range(MAX_ITER):
		feed_d = {input_placeholder:x, label_placeholder:y}
		ls, _, ac = sess.run([loss,train_step,accuracy],feed_dict = feed_d)
		if i%100==0:
			print('Loss at Iter',i,':',ls,'\tacc:',ac)