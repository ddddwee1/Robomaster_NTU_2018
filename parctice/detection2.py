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
		f = open('annot_reform2.txt')
		for i in f:
			if 'nan' in i:
				continue
			i = i.strip()
			i = i.split('\t')
			img = cv2.imread(i[0])
			print(i[0])
			img = cv2.resize(img,(256,256))
			i = i[1:]
			i = [float(k) for k in i]
			if i[0]>=240:
				i[0] = 239
			if i[1]>=180:
				i[1] = 179
			if i[2]>=240:
				i[2] = 239
			if i[3]>=180:
				i[3] = 179
			data.append([img,[[float(i[0])*256/240,float(i[1])*256/180,float(i[2])*128/240,float(i[3])*128/180]]])
		self.data = data

	def get_conf_mtx(self,inp):
		conf_res = np.zeros([16,16,1])
		for i in range(len(inp)):
			x = inp[i][0]
			y = inp[i][1]
			col = int(np.floor(x/16))
			row = int(np.floor(y/16))
			conf_res[row][col][0] = 1
		return conf_res

	def get_bbox_mtx(self,inp):
		bbox_res = np.zeros([16,16,4],dtype=np.float32)
		for i in range(len(inp)):
			x = inp[i][0]
			y = inp[i][1]
			w = inp[i][2]
			h = inp[i][3]
			# print(x,y)
			col = int(np.floor(x/16))
			row = int(np.floor(y/16))
			bbox_res[row][col][0] = x - col*16.0 - 8.0
			bbox_res[row][col][1] = y - row*16.0 - 8.0
			bbox_res[row][col][2] = w
			bbox_res[row][col][3] = h
		return bbox_res

	def random_trans_img(self,img,lmk):
		while True:
			scale = np.random.rand()
			# scale = 0.85 + scale/4
			# print(scale)
			scale = 1.0
			lmk = np.float32(lmk).copy()
			lmk = lmk * scale
			scale_hw = int(256*scale)
			x_low = int(25.-lmk[0][0])
			x_high = int(235.-lmk[0][0])
			y_low = int(25.-lmk[0][1])
			y_high = int(235.-lmk[0][1])
			if x_high>x_low and y_high>y_low:
				break
		img = cv2.resize(img,(scale_hw,scale_hw))
		shift_x = np.random.randint(x_low,x_high)
		shift_y = np.random.randint(y_low,y_high)
		M = np.float32([[1,0,shift_x],[0,1,shift_y]])
		img = cv2.warpAffine(img,M,(256,256))
		lmk = np.float32(lmk).copy()
		lmk += np.float32([[shift_x,shift_y,0,0]])
		return img,lmk

	def next_train_batch(self,bsize):
		batch = random.sample(self.data,bsize)
		a = []
		for i in batch:
			img,lmk = self.random_trans_img(i[0],i[1])
			a.append([img,self.get_bbox_mtx(lmk),self.get_conf_mtx(lmk)])
		return a

MAXITER = 50000*2
BSIZE = 32

def draw(img,c,b):
	print(c.shape)
	print(b.shape)
	print(c.max())
	for i in range(16):
		for j in range(16):
			if c[i][j][0]>0:
				x = int(b[i][j][0])+j*16+8
				y = int(b[i][j][1])+i*16+8
				w = int(b[i][j][2])
				h = int(b[i][j][3])
				print(b[i][j])
				# cv2.circle(img,(x,y),5,(0,0,255),-1)
				cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


imgholder,biasholder,confholder,bias_loss,conf_loss,train_step,conf,bias = build_graph()
with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	saver = tf.train.Saver()
	reader = data_reader()
	print('Reading finish')
	for iteration in range(MAXITER):
		# print(iteration)
		train_batch = reader.next_train_batch(BSIZE)
		# print(iteration)
		img_batch = [i[0] for i in train_batch]
		bias_batch = [i[1] for i in train_batch]
		conf_batch = [i[2] for i in train_batch]
		img_batch = np.float32(img_batch)
		feeddict = {imgholder:img_batch, biasholder:bias_batch, confholder:conf_batch}
		# print(iteration)
		b_loss, c_loss, _, c,b = sess.run([bias_loss,conf_loss,train_step,conf,bias],feed_dict=feeddict)

		if iteration%10==0:
			print('Iter:',iteration,'\tLoss_b:',b_loss,'\tLoss_c:',c_loss)
			# print(c.max())
			# img = img_batch[0].astype(np.uint8)
			# draw(img,c[0],b[0])
			# draw(img_batch[0],conf_batch[0],bias_batch[0])
 
		if iteration%5000==0 and iteration!=0:
			saver.save(sess,'./model/'+str(iteration)+'.ckpt')