import tensorflow as tf 
import model as M 
import numpy as np 

BSIZE = 128

def build_model(img_input):
	# can change whatever activation function
	with tf.variable_scope('mnist'):
		mod = M.Model(img_input,[None,28*28])
		mod.reshape([-1,28,28,1])
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.flatten()
		mod.fcLayer(50,activation=M.PARAM_LRELU)
		mod.fcLayer(10)
	return mod.get_current_layer()

def build_7seg_model(img_input):
	with tf.variable_scope('7seg_detection'):
		mod = M.Model(img_input,[None,28*28])
		mod.reshape([-1,28,28,1])
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.flatten()
		mod.fcLayer(50,activation=M.PARAM_LRELU)
		mod.fcLayer(10)
	return mod.get_current_layer()

def build_graph():
	img_holder = tf.placeholder(tf.float32,[None,28*28])
	last_layer_7seg = build_7seg_model(img_holder)
	last_layer = build_model(img_holder)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=img_holder,logits=last_layer))
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=img_holder,logits=last_layer))
	accuracy = M.accuracy(last_layer,tf.argmax(lab_holder,1))
	train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
	return img_holder,last_layer,last_layer_7seg

img_holder,last_layer,last_layer_7seg = build_graph()

sess = tf.Session()
M.loadSess('./model_mnist/',sess,var_list=M.get_all_vars('mnist'))
M.loadSess('./model_7seg/',sess,var_list=M.get_all_vars('7seg_detection'))

def get_pred(imgs):
	scr = sess.run(tf.argmax(last_layer,1),feed_dict={img_holder:imgs})
	return scr

def get_pred_7seg(imgs):
	scr = sess.run(tf.argmax(last_layer_7seg,1),feed_dict={img_holder:imgs})
	return scr 
