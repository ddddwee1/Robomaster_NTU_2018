import cv2
import model as M 
import tensorflow as tf 
import numpy as np 
import random

# f = open('trainlist.txt')
# for line in f:
# 	line = line.strip()
# 	line_data = line.split('\t')
# 	image_path = line_data[0]
# 	img = cv2.imread(image_path)
# 	img = cv2.resize(img,(128,128))
# 	print(img.shape)
# 	cv2.imshow('Image',img)
# 	cv2.waitKey(0)
# 	print(line_data)
# 	# input('Pause')

def get_data():
	data = []

	f = open('trainlist.txt')
	for line in f:
		if 'nan' in line:
			continue
		line = line.strip()
		line_data = line.split('\t')
		image_path = line_data[0]
		img = cv2.imread(image_path,0)
		# img = img.reshape([96,96,1])
		img = cv2.resize(img,(128,128))
		img = img.reshape([128,128,1])
		x1 = float(line_data[1])
		y1 = float(line_data[2])
		x2 = float(line_data[3])
		y2 = float(line_data[4])
		data_row = [img,[x1,y1,x2,y2]]
		data.append(data_row)

	return data

def build_model(input_placeholder):
	mod = M.Model(input_placeholder,[None,128,128,1])
	mod.convLayer(5,16,stride=2,activation=M.PARAM_RELU)
	mod.convLayer(4,32,stride=2,activation=M.PARAM_RELU)
	mod.convLayer(3,32,stride=2,activation=M.PARAM_RELU)
	mod.convLayer(3,64,activation=M.PARAM_RELU)
	mod.convLayer(3,64,stride=2,activation=M.PARAM_RELU)
	mod.convLayer(3,128,activation=M.PARAM_RELU)
	mod.convLayer(3,128,stride=2,activation=M.PARAM_RELU)
	#copy the model
	mod.flatten()
	mod.fcLayer(100,activation=M.PARAM_RELU)
	mod.fcLayer(4)
	return mod.get_current_layer()

def build_graph():
	img_placeholder = tf.placeholder(tf.float32,[None,128,128,1])
	lab_placeholder = tf.placeholder(tf.float32,[None,4])
	lr_placeholder = tf.placeholder(tf.float32)
	output = build_model(img_placeholder)
	loss = tf.reduce_mean(tf.square(output - lab_placeholder))
	train_step = tf.train.AdamOptimizer(lr_placeholder).minimize(loss)
	return img_placeholder,lab_placeholder,output,loss,train_step,lr_placeholder

MAX_ITER = 20000
BSIZE = 32 #16
img_placeholder,lab_placeholder,output,loss,train_step,lr_placeholder = build_graph()
data = get_data()

with tf.Session() as sess:
	saver = tf.train.Saver()
	M.loadSess('./model/',sess,init=True)
	for iteration in range(MAX_ITER):
		databatch = random.sample(data,BSIZE)
		img_batch = [i[0] for i in databatch]
		xy_batch = [i[1] for i in databatch]
		# print(xy_batch[0])
		# input()
		feed_d = {img_placeholder:img_batch, lab_placeholder: xy_batch, lr_placeholder: 0.001}
		ls, _ = sess.run([loss,train_step],feed_dict=feed_d)
		out = sess.run(output,feed_dict={img_placeholder:out_img})
		#out: [1,4]
		print(ls)

		if iteration%100==0:
			print('Iter:',iteration,'\tLoss:',ls)

		if iteration%2000==0 and iteration!=0:
			saver.save(sess,'./model/'+str(iteration)+'.ckpt')