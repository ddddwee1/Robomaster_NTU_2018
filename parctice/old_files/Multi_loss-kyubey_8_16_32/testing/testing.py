import tensorflow as tf
import graph
from data_reader1 import data_reader
import model as M
import numpy as np 
import cv2
import Functions as F

def draw(img,c,b,multip):
	row,col,_ = b.shape
	# print(b.shape,c.shape)
	# print(row,col)
	for i in range(row):
		for j in range(col):
			# print(i,j)
			if c[i][j][0]==c.max():
				x = int(b[i][j][0])+j*multip+multip//2
				y = int(b[i][j][1])+i*multip+multip//2
				w = int(b[i][j][2])
				h = int(b[i][j][3])
				cv2.rectangle(img,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,0,255),2)

def draw2(img,c,b,multip):
	row,col,_ = b.shape
	c = c.reshape([-1])
	ind = c.argsort()[-3:][::-1]
	for aaa in ind:
		# print(aaa)
		i = aaa//col
		j = aaa%col 
		x = int(b[i][j][0])+j*multip+multip//2
		y = int(b[i][j][1])+i*multip+multip//2
		w = int(b[i][j][2])
		h = int(b[i][j][3])
		cv2.rectangle(img,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,255,0),2)


def train_RPN(imgholder,RPNcb):
	MAXITER = 1000000000

	#config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.2
	#sess = tf.Session(config=tf.ConfigProto( allow_soft_placement=True, log_device_placement=True))
	with tf.Session() as sess:
		M.loadSess('./model_RPN/',sess,init=True)
		saver = tf.train.Saver()
		reader = data_reader('2018-03-17-164409.webm')
		size = reader.get_size()
		print('Reading finish')
		for i in range(size):
			img = reader.get_img(i)
			img1 = np.float32(img).copy()
			img1 = np.uint8(img1)
			img2 = np.float32(img).copy()
			img2 = np.uint8(img2)
			img3 = np.float32(img).copy()
			img3 = np.uint8(img3)

			b0,b1,b2,c0,c1,c2= sess.run([RPNcb[0],RPNcb[1],RPNcb[2],RPNcb[3],RPNcb[4],RPNcb[5]],feed_dict={imgholder:[img]})
			print ( c0.max() , c1.max() , c2.max())
			draw2(img1,c0[0],b0[0],8)
			draw(img1,c0[0],b0[0],8)
			img1=cv2.resize(img1,(480,270))
			cv2.imshow('img',img1)

			draw2(img2,c1[0],b1[0],16)
			draw(img2,c1[0],b1[0],16)
			img2=cv2.resize(img2,(480,270))
			cv2.imshow('img1',img2)

			draw2(img3,c2[0],b2[0],32)
			draw(img3,c2[0],b2[0],32)
			img3=cv2.resize(img3,(480,270))
			cv2.imshow('img2',img3)

			cv2.waitKey(100)


def train_veri(imgholder,RPNcb,croppedholder,veri_conf_holder,veri_conf_loss,veri_train_step,veri_conf,veri_accuracy):
	MAXITER = 50000
#	MAXITER = 1
	BSIZE = 1
	with tf.Session() as sess:
		M.loadSess('drive/Colab/veri_original_image/training/model_VERI/',sess,init=True)
		M.loadSess(modpath='drive/Colab/veri_original_image/training/model_RPN/50000.ckpt',sess=sess,var_list=M.get_trainable_vars('mainModel'))
		saver = tf.train.Saver()
		reader = data_reader('drive/Colab/veri_original_image/training/annotationC.txt')
		print('Reading finish')
		for iteration in range(MAXITER):
			train_batch,nscale = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch0 = [i[1] for i in train_batch]
			conf_batch0 = [i[2] for i in train_batch]
			bias_batch2 = [i[3] for i in train_batch]
			conf_batch2 = [i[4] for i in train_batch]

			img_batch = np.float32(img_batch)

			feeddict1 = {imgholder:img_batch}	# Running RPN (already trained)
			b0,b2,c0,c2= sess.run([RPNcb[0],RPNcb[1],RPNcb[2],RPNcb[3]],feed_dict=feeddict1)	# getting c,b output from the RPN
			cropped_batch = []
			veri_conf_batch = []
			for i in range(len(img_batch)):
				croppedImages, labels = F.dual_crop_original(img_batch[i], bias_batch0[i], conf_batch0[i],bias_batch2[i], conf_batch2[i], b0[i],b2[i],c0[i],c2[i])
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
			if iteration%5000==0 and iteration!=0:
				saver.save(sess,'drive/Colab/veri_original_image/training/model_VERI/'+str(iteration)+'.ckpt')



create_graph=graph.build_graph()
RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb = create_graph.graphs()
#croppedholder, veri_conf_holder, veri_conf_loss, veri_train_step, veri_conf, veri_accuracy = build_graph2()

## Train RPN

train_RPN(RPNholders[0],RPNcb)

## Train verification

#train_veri(RPNholders[0],RPNcb[0],RPNcb[1],croppedholder, veri_conf_holder, veri_conf_loss, veri_train_step, veri_conf, veri_accuracy)
#train_veri(RPNholders[0],RPNcb,veriholders[0],veriholders[1],verilosses[0],train_steps[1],vericb[0],vericb[1])
