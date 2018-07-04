import tensorflow as tf 
from net import model as M 
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
		mod.fcLayer(11)
	return mod.get_current_layer()

def build_FD_model(img_input):
	with tf.variable_scope('FD_detection'):
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
		mod.fcLayer(11)
	return mod.get_current_layer()

def build_graph():
	img_holder = tf.placeholder(tf.float32,[None,28*28])
	last_layer = build_model(img_holder)
	last_layer_7seg = build_7seg_model(img_holder)
	last_layer_FD = build_FD_model(img_holder)
	return img_holder,last_layer,last_layer_7seg,last_layer_FD

img_holder,last_layer,last_layer_7seg,last_layer_FD = build_graph()

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
M.loadSess('./model_rune/model_mnist/',sess,var_list=M.get_all_vars('mnist'))
M.loadSess('./model_rune/model_7seg/',sess,var_list=M.get_all_vars('7seg_detection'))
M.loadSess('./model_rune/model_flaming/',sess,var_list=M.get_all_vars('FD_detection'))

def get_pred(imgs):
	scr = sess.run(tf.argmax(last_layer,1),feed_dict={img_holder:imgs})
	return scr

def get_pred_7seg(imgs):
	scr = sess.run(tf.argmax(last_layer_7seg,1),feed_dict={img_holder:imgs})
	return scr 

def get_pred_flaming(imgs):
	scr = sess.run(tf.argmax(last_layer_FD,1),feed_dict={img_holder:imgs})
	return scr 
