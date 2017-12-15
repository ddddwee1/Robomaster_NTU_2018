import tensorflow as tf
import graph
from data_reader import data_reader
import model as M
import numpy as np 
import cv2
import Functions as F

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


def crop_original(img,bias_mtx,conf_mtx,c,b):
	"""
	Input: Original Image, Confidence matrix and Bias matrix
	Output: Array of 5 cropped images(size 32x32)
	Description: Select the top 5 regions with highest confidence and return cropped images of those regions  
	"""
	croppedImages = []
	labels = []
	indices = np.argsort(c,axis=None)      #we get the indices of flattened array in descending order
	for a in range(-1,-6,-1):
		ind = indices[a]
		i = ind//16                 
		j = ind%16
#		print('a:',a,'\tind:',ind,'\ti:',i,'\tj:',j,'\tconf:',c[i][j],'\txbias',int(b[i][j][0]),'\tybias',int(b[i][j][1]))
		# cropping:
		x = int(b[i][j][0])+j*16+8           #x-coord of btm right of a the object center wrt to original image(256x256)
		y = int(b[i][j][1])+i*16+8           #y-coord of btm right of a the object center wrt to original image(256x256)
		w = int(b[i][j][2])                  #half of width of the object
		h = int(b[i][j][3])				     #half of height of the object
		x = x-w                              #x-coord of center of a the object center wrt to original image(256x256)
		y = y-h                              #y-coord of center of a the object center wrt to original image(256x256)
#		print('x:',x,'\ty:',y,'\tw:',w,'\th:',h)
		x_low = x-w
		x_high = x+w
		y_low = y-h
		y_high = y+h
		if x_low<0:
			x_low=0
		elif x_high>255:
			x_high=255
		if y_low<0:
			y_low=0
		elif y_high>255:
			y_high=255
		cp = img[y_low:y_high,x_low:x_high]
		cp = cv2.resize(cp,(32,32))
		croppedImages.append(cp)

		# labelling:

		inp1 = [x, y, w, h]
		for r in range(16):
			for c in range(16):
				if conf_mtx[r][c][0] == 1:
					inp2 = bias_mtx[r][c].copy()
					# print("r:", r, "\tc:", c, "\t", inp2)
					inp2[0] = inp2[0]+c*16+8-inp2[2]
					inp2[1] = inp2[1]+r*16+8-inp2[3]
					break

		iou = F.get_iou(inp1, inp2)
		# print("inp1:", inp1, "\tinp2:", inp2, "\tiou", iou)
		if iou > 0.3:
			label = 1
		else:
			label = 0
		labels.append(label)

	return croppedImages,labels
	
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

def train_veri(imgholder,conf,bias,croppedholder,veri_conf_holder,veri_conf_loss,veri_train_step,veri_conf,veri_accuracy):
	MAXITER = 50000
#	MAXITER = 1
	BSIZE = 40
	with tf.Session() as sess:
		M.loadSess('./model_VERI/',sess,init=True)
		M.loadSess(modpath='./model_RPN/20000.ckpt',sess=sess,var_list=M.get_trainable_vars('mainModel'))
		saver = tf.train.Saver()
		reader = data_reader('merge.txt')
		print('Reading finish')
		for iteration in range(MAXITER):
			train_batch = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch = [i[1] for i in train_batch]
			conf_batch = [i[2] for i in train_batch]
#			mask_batch = [i[3] for i in train_batch]
			img_batch = np.float32(img_batch)
			feeddict1 = {imgholder:img_batch}	# Running RPN (already trained)
			c, b = sess.run([conf, bias],feed_dict=feeddict1)	# getting c,b output from the RPN
			cropped_batch = []
			veri_conf_batch = []
			for i in range(len(img_batch)):
				croppedImages, labels = crop_original(img_batch[i], bias_batch[i], conf_batch[i], c[i], b[i])
				for j in range(len(labels)):
					cropped_batch.append(croppedImages[j])
					veri_conf_batch.append([labels[j]])
			# truenums = [item[0] for item in veri_conf_batch]
			# truenums = sum(truenums)
			# print(truenums/200)
			feeddict2 = {croppedholder:cropped_batch, veri_conf_holder:veri_conf_batch}
			loss, _, acc = sess.run([veri_conf_loss, veri_train_step, veri_accuracy], feed_dict=feeddict2)
			#Testing cropping
			# for i in range(len(cropped_batch)):
			# 	img = cropped_batch[i].astype(np.uint8)
			# 	print("label:\t", veri_conf_batch[i])
			# 	cv2.imshow('img',img)
			# 	cv2.waitKey(10)
			# 	inp=input()
			if iteration%10==0:
				print('Iter:',iteration,'\tLoss:',loss,'\taccuracy:',acc)		
			if iteration%1000==0 and iteration!=0:
				saver.save(sess,'./model_VERI/'+str(iteration)+'.ckpt')


RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb, feature_map = graph.build_graph()
#croppedholder, veri_conf_holder, veri_conf_loss, veri_train_step, veri_conf, veri_accuracy = build_graph2()

## Train RPN

#train_RPN(RPNholders[0],RPNholders[1],RPNholders[2], RPNholders[3], RPNlosses[0], RPNlosses[1], train_steps[0], RPNcb[0], RPNcb[1])

## Train verification

#train_veri(RPNholders[0],RPNcb[0],RPNcb[1],croppedholder, veri_conf_holder, veri_conf_loss, veri_train_step, veri_conf, veri_accuracy)
train_veri(RPNholders[0],RPNcb[0],RPNcb[1],veriholders[0],veriholders[1],verilosses[0],train_steps[1],vericb[0],vericb[1])
