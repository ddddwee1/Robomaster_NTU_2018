import tensorflow as tf 
import model as M 
import numpy as np 

BSIZE = 128

def build_model(img_input):
	# can change whatever activation function
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
	lab_holder = tf.placeholder(tf.float32,[None,10])
	last_layer = build_model(img_holder)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=lab_holder,logits=last_layer))
	accuracy = M.accuracy(last_layer,tf.argmax(lab_holder,1))
	train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
	return img_holder,lab_holder,loss,train_step,accuracy,last_layer

img_holder,lab_holder,loss,train_step,accuracy,last_layer = build_graph()

config - tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
M.loadSess('./model/',sess)

def get_pred(imgs):
	scr = sess.run(tf.argmax(last_layer,1),feed_dict={img_holder:imgs})
	return scr
