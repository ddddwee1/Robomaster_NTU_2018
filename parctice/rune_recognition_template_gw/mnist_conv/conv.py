import tensorflow as tf 
import model as M 
import numpy as np 
import cv2
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
BSIZE = 128
class data_reader():

	def __init__(self):
		print('Reading data...')

	def next_train_batch(self,x):
		self.x_train=[]
		self.y_train=[]
		x= np.array(x)
		x = x.reshape([-1,28,28])
		#print (x.shape)
		for i in range(x.shape[0]):
			buff = x[i]
			buff = buff*255
			buff = 255-buff
			buff = np.uint8(buff)
			edged = cv2.Canny(buff, 50, 255, 255)
			kernel = np.ones((2,2),np.uint8)
			edged = edged.reshape([784])
			self.x_train.append(edged)
			#cv_image_gray = cv2.cvtColor(buff, cv2.COLOR_BGR2GRAY)
			#ret,cv_image_binary = cv2.threshold(cv_image_gray,128,255,cv2.THRESH_BINARY_INV)
			#buff_scaled = cv2.resize(buff,None,fx = 1, fy = 1, interpolation = cv2.INTER_LINEAR)
			#buff = buff.reshape([-1,28,28])
			#print (buff.shape)

			#x_rand = np.random.randint(260)
			#y_rand = np.random.randint(180)

			#M = np.float32([[1,0,x_rand],[0,1,y_rand]])
			#img_result = cv2.warpAffine(buff,M,(320,240))
			#self.x_train = np.append(self.x_train,img_result)
			#self.x_train = self.x_train.reshape([-1,320,240,1])

			#self.conf_mtx = self.get_conf_mtx(y[i],x_rand,y_rand)
			#self.y_train = np.append(self.y_train,self.conf_mtx)
			#self.y_train = self.y_train.reshape([-1,20,15,10])

			#print (self.y_train.shape,'abc')
			#print (x_rand,y_rand)
			#cv2.imshow('img', edged)
			#cv2.waitKey(0)

		return self.x_train


def build_model(img_input):
	# can change whatever activation function
	mod = M.Model(img_input,[None,28*28])
	mod.reshape([-1,28,28,1])
	mod.convLayer(5,16,activation=M.PARAM_LRELU)
	mod.dropout(0.8)
	mod.maxpoolLayer(2)
	mod.convLayer(5,32,activation=M.PARAM_LRELU)
	mod.dropout(0.9)
	mod.maxpoolLayer(2)
	mod.convLayer(3,48,activation=M.PARAM_LRELU)
	mod.dropout(0.9)
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
	train_step = tf.train.AdamOptimizer(0.000001).minimize(loss)
	return img_holder,lab_holder,loss,train_step,accuracy,last_layer

img_holder,lab_holder,loss,train_step,accuracy,last_layer = build_graph()

with tf.Session() as sess:
	saver = tf.train.Saver()
	reader = data_reader()
	M.loadSess('./model/',sess,init=True)
	for i in range(10000000):
		x, y_train = mnist.train.next_batch(BSIZE)
		x_train = reader.next_train_batch(x)
		_,acc,ls = sess.run([train_step,accuracy,loss],feed_dict={img_holder:x_train,lab_holder:y_train})
		if i%100==0:
			print('iter',i,'\t|acc:',acc,'\tloss:',ls)
		if i%1000==0:
			x1, y_test = mnist.test.next_batch(64)
			x_test = reader.next_train_batch(x1)
			acc = sess.run(accuracy,feed_dict={img_holder:x_test, lab_holder:y_test})
			print('Test accuracy:',acc)
		if i%1000==0 and i!=0:
			saver.save(sess,'./model/'+str(i)+'.ckpt')


# -------------------------
# Run time testing
# -------------------------


# import time
# with tf.Session() as sess:
# 	M.loadSess('./model/',sess,init=True)
# 	for i in range(1500):
# 		x = np.ones([9,28*28]).astype(np.float32)
# 		sess.run(last_layer,feed_dict={img_holder:x})
# 		if i==500:
# 			timea = time.time()
# 	timeb = time.time()
# 	print('Time elapsed:', timeb - timea)
