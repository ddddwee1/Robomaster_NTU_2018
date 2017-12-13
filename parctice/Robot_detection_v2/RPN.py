import tensorflow as tf 
import numpy as np 
import model as M 
import random
import cv2

def main_structure(inp):
	with tf.variable_scope('mainModel'):
		inp = tf.scan(lambda _,y:tf.image.random_brightness(y,20),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
		inp = tf.scan(lambda _,y:tf.image.random_contrast(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
		inp = tf.scan(lambda _,y:tf.image.random_saturation(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
		mod = M.Model(inp,[None,256,256,3])
		mod.convLayer(5,16,stride=2,activation=M.PARAM_RELU)#128_2x2
		mod.convLayer(4,32,stride=2,activation=M.PARAM_RELU)#64_4x4
		mod.convLayer(3,64,stride=2,activation=M.PARAM_MFM)#32_8x8
		mod.convLayer(3,128,activation=M.PARAM_MFM)#32_8x8
		mod.convLayer(3,256,stride=2,activation=M.PARAM_RELU)#16_16x16
		mod.convLayer(3,256,activation=M.PARAM_RELU)
		mod.convLayer(3,256,activation=M.PARAM_RELU)
		return mod.get_current_layer()

def detection_parts(inp):
	with tf.variable_scope('detectionLayers'):
		mod = M.Model(inp,[None,16,16,128])
		feature =mod.convLayer(1,128*2,activation=M.PARAM_RELU)
		bias = mod.convLayer(1,4)  #x_bias, y_bias
		mod.set_current(feature)
		conf = mod.convLayer(1,1)  #confidence of object
		mod.reshape([-1,16,16])
		return conf[0],bias[0]
