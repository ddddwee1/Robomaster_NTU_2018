import tensorflow as tf
import graph
from data_reader import data_reader
import model as M
import numpy as np 
import cv2

def draw(img,c,b,wait=0):
	for i in range(16):
		for j in range(16):
			if c[i][j][0]>0:
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

def train_RPN(imgholder,biasholder,confholder,maskholder,bias_loss,conf_loss,train_step,conf,bias):
	MAXITER = 50000*2
	BSIZE = 32
	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=True)
		saver = tf.train.Saver()
		reader = data_reader('merge.txt')
		print('Reading finish')
		for iteration in range(MAXITER):
			train_batch = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch = [i[1] for i in train_batch]
			conf_batch = [i[2] for i in train_batch]
			mask_batch = [i[3] for i in train_batch]
			img_batch = np.float32(img_batch)
			feeddict = {imgholder:img_batch, biasholder:bias_batch, confholder:conf_batch, maskholder:mask_batch}
			b_loss, c_loss, _, c, b	= sess.run([bias_loss,conf_loss,train_step,conf, bias],feed_dict=feeddict)
			if iteration%10==0:
				print('Iter:',iteration,'\tLoss_b:',b_loss,'\tLoss_c:',c_loss)		
				img = img_batch[0].astype(np.uint8)
				draw(img,c[0],b[0],wait=5)
				# draw(img,conf_batch[0],bias_batch[0],wait=5)
			if iteration%5000==0 and iteration!=0:
				saver.save(sess,'./model/'+str(iteration)+'.ckpt')

RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb, feature_map = graph.build_graph()
train_RPN(RPNholders[0],RPNholders[1],RPNholders[2], RPNholders[3], RPNlosses[0], RPNlosses[1], train_steps[0], RPNcb[0], RPNcb[1])