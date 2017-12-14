import tensorflow as tf 
import numpy as np 
import model as M 
import random
import cv2

def main_structure(inp):
	with tf.variable_scope('mainModel'):
		mod = M.Model(inp,[None,256,256,3])
		mod.convLayer(5,16,stride=2,activation=M.PARAM_RELU)#128_2x2
		mod.convLayer(4,32,stride=2,activation=M.PARAM_RELU)#64_4x4
		mod.convLayer(3,64,stride=2,activation=M.PARAM_MFM)#32_8x8
		mod.convLayer(3,128,activation=M.PARAM_MFM)#32_8x8
		mod.convLayer(3,256,stride=2,activation=M.PARAM_MFM)#16_16x16
		mod.convLayer(3,256,activation=M.PARAM_MFM)
		mod.convLayer(3,256,activation=M.PARAM_MFM)
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

def build_graph():
	with tf.name_scope('imgholder'):
		imgholder = tf.placeholder(tf.float32,[None,256,256,3])
	with tf.name_scope('biasholder'):
		biasholder = tf.placeholder(tf.float32,[None,16,16,4])
	with tf.name_scope('confidence'):
		confholder = tf.placeholder(tf.float32,[None,16,16,1])

	feature_map = main_structure(imgholder)
	conf, bias = detection_parts(feature_map)

	bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(bias*confholder - biasholder),axis=0))
	conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conf,labels=confholder))

	train_step = tf.train.AdamOptimizer(0.00001).minimize(bias_loss+conf_loss)

	return imgholder,biasholder,confholder,bias_loss,conf_loss,train_step,conf,bias

class data_reader():
	def __init__(self):
		print('Reading data...')
		data = []
		f = open('test_list.txt')
		for i in f:
			i = i.strip()
			i = i.split('\t')
			img = cv2.imread(i[0])
			print(i[0])
			img = cv2.resize(img,(256,256))
			data.append([img,[0]])
		self.data = data

	def get_i(self,i):
		return [self.data[i]]

	def get_j(self):
		return len(self.data)

def draw(img,c,b):
	print(c.shape)
	print(b.shape)
	print(c.max())
	for i in range(16):
		for j in range(16):
			if c[i][j][0]>-0.2:
				x = int(b[i][j][0])+j*16+8
				y = int(b[i][j][1])+i*16+8
				w = int(b[i][j][2])
				h = int(b[i][j][3])
				print(b[i][j])
				# cv2.circle(img,(x,y),5,(0,0,255),-1)
				cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(5)
	# cv2.destroyAllWindows()


imgholder,biasholder,confholder,bias_loss,conf_loss,train_step,conf,bias = build_graph()
with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	reader = data_reader()
	MAXITER = reader.get_j()
	print('Reading finish')
	for iteration in range(MAXITER):
		train_batch = reader.get_i(iteration)
		img_batch = [i[0] for i in train_batch]
		feeddict = {imgholder:img_batch}
		c,b = sess.run([conf,bias],feed_dict=feeddict)

		if iteration%10==0:
			print(c.max())
			img = img_batch[0].astype(np.uint8)
			draw(img,c[0],b[0])

		if iteration%5000==0 and iteration!=0:
			saver.save(sess,'./model/'+str(iteration)+'.ckpt')