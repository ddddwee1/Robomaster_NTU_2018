import tensorflow as tf
import graph
from data_reader import data_reader_test as data_reader
import model as M
import numpy as np 
import cv2

def draw(img,c,b,wait=0):
	for i in range(16):
		for j in range(16):
			if c[i][j][0]>-0.5:
				x = int(b[i][j][0])+j*16+8
				y = int(b[i][j][1])+i*16+8
				w = int(b[i][j][2])
				h = int(b[i][j][3])
				x = x-w
				y = y-h
				cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(wait)
	# cv2.destroyAllWindows()

def draw2(img,c,b,wait=1):
	ind = np.argmax(c)
	i = ind//16
	j = ind%16
	x = int(b[i][j][0])+j*16+8
	y = int(b[i][j][1])+i*16+8
	w = int(b[i][j][2])
	h = int(b[i][j][3])
	x = x-w
	y = y-h
	cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(wait)
	# cv2.destroyAllWindows()

def test_RPN(imgholder,biasholder,confholder,maskholder,bias_loss,conf_loss,train_step,conf,bias):
	with tf.Session() as sess:
		M.loadSess('',sess,init=True)
		M.loadSess('./model/',sess,init=True,var_list=M.get_trainable_vars('mainModel'))
		saver = tf.train.Saver()
		reader = data_reader('./video/vid1.mp4')
		print('Reading finish')
		for iteration in range(reader.get_iter()):
			img_batch = reader.next_img(iteration)
			img_batch = np.float32(img_batch)
			feeddict = {imgholder:img_batch}
			c, b = sess.run([conf, bias],feed_dict=feeddict)
			img = img_batch[0].astype(np.uint8)
			draw2(img,c[0],b[0],wait=10)

RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb, feature_map = graph.build_graph(test=True)
test_RPN(RPNholders[0],RPNholders[1],RPNholders[2], RPNholders[3], RPNlosses[0], RPNlosses[1], train_steps[0], RPNcb[0], RPNcb[1])