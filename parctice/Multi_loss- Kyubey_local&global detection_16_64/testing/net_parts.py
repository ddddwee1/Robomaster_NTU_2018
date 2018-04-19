import tensorflow as tf 
import numpy as np 
import model as M 
import cv2 


def verify_net(inp,test):
	with tf.variable_scope('detectionLayers'):
		if not test:
			inp = tf.scan(lambda _,y:tf.image.random_brightness(y,20),inp,initializer=tf.constant(0.0,shape=[16,16,3]))
			inp = tf.scan(lambda _,y:tf.image.random_contrast(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[16,16,3]))
			inp = tf.scan(lambda _,y:tf.image.random_saturation(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[16,16,3]))
		mod = M.Model(inp,[None,16,16,3])
		mod.convLayer(3,16,stride=1,activation=M.PARAM_RELU)		#32_8x8
		mod.maxpoolLayer(2)
		mod.convLayer(3,32,stride=1,activation=M.PARAM_RELU)	#16_16x16
		mod.maxpoolLayer(2)
		mod.convLayer(2,48,stride=1,activation=M.PARAM_RELU)	#8_16x16	
		mod.maxpoolLayer(2)
		mod.flatten()
		mod.fcLayer(200,activation=M.PARAM_RELU)
		mod.fcLayer(1)
		return mod.get_current_layer()

def multiloss_RPN(inp,test):
	with tf.variable_scope('mainModel'):
		if not test:
			inp = tf.scan(lambda _,y:tf.image.random_brightness(y,20),inp)
			inp = tf.scan(lambda _,y:tf.image.random_contrast(y,0.5,2),inp)
			inp = tf.scan(lambda _,y:tf.image.random_saturation(y,0.5,2),inp)

		mod = M.Model(inp,[None,None,None,3])
		mod.dwconvLayer(7,5,stride=2,activation=M.PARAM_LRELU)
		mod.convLayer(2,16,activation=M.PARAM_LRELU) #480_ 2x2 channel merging
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU) #240_ 4x4 can add layers to change the size of detection grid
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU)#120_ 8x8

		c0 =mod.convLayer(4,32,stride=2,activation=M.PARAM_LRELU) #60_ 16x16
		mod.convLayer(3,64,activation=M.PARAM_LRELU)
		feat0 = mod.convLayer(3,128,activation=M.PARAM_LRELU)
		bias0 = mod.convLayer(1,4)
		mod.set_current(feat0)
		conf0 = mod.convLayer(1,1)

		mod.set_current(c0)
		mod.convLayer(4,64,stride=2,activation=M.PARAM_LRELU) #30_ 32x32
		#mod.convLayer(4,48,activation=M.PARAM_LRELU)
		#feat1 = mod.convLayer(3,64,activation=M.PARAM_LRELU)
		#bias1 = mod.convLayer(1,4)
		#mod.set_current(feat1)
		#conf1 = mod.convLayer(1,1)

		#mod.set_current(c1)
		c2 = mod.convLayer(3,64,stride=2,activation=M.PARAM_LRELU) #15_ 64x64
		mod.convLayer(3,64,activation=M.PARAM_LRELU)
		feat2 = mod.convLayer(3,128,activation=M.PARAM_LRELU)
		bias2 = mod.convLayer(1,4)
		mod.set_current(feat2)
		conf2 = mod.convLayer(1,1)


		return bias0[0],bias2[0],conf0[0],conf2[0]


#loss_functions = build_loss(inpholder,b_labholder,c_labholder)
#train_steps = get_train_steps(loss_functions)

# 100x100
# i decide to use the index = 1
# _,loss = sees.run([train_steps[index],loss_functions[index]],feed_dict={sdfasdfsadfsadf})
# print(loss)

