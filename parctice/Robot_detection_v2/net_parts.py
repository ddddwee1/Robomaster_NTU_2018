import tensorflow as tf 
import numpy as np 
import model as M 
import cv2 

def verify_net(inp):
	with tf.variable_scope('detectionLayers'):
		mod = M.Model(inp,[None,3,3,256])
		# 3 * 3 size of feature map; 256 refers to the number of maps
		mod.convlayer(1,256,activation = M.PARAM_RELU)
		# 1 refers to the kernal size. 256 refers to the number of feature maps.
		mod.convlayer(1,256,activation = M.PARAM_RELU)
		mod.flatten()		
		feature = mod.fcLayer(256,activation = M.PARAM_RELU)
		# 256 refers to the number of neurals
		veri_bias = mod.fcLayer(4)
		mod.set_current(feature)
		veri_conf= mod.fcLayer(1)  # [Layer, Shape]
	return veri_conf[0],veri_bias[0] 
