import tensorflow as tf
import graph
from data_reader import data_reader
import model as M
import numpy as np 
import cv2
import Functions as F

def draw(img,c,b,scale,wait=0):
	# print(c.shape)
	# print(b.shape)
	# print(c.max())
	for i in range(int(34*scale)):
		for j in range(int(60*scale)):
			if c[i][j][0] == c.max():
				x = int(b[i][j][0]+j*32/scale+16/scale)
				y = int(b[i][j][1]+i*32/scale+16/scale)
				w = int(b[i][j][2])
				h = int(b[i][j][3])
				print (c.max())
				# print(b[i][j])
				# cv2.circle(img,(x,y),5,(0,0,255),-1)
				cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,0,255),2)

			else:
				if c[i][j][0]>0.0:
					x = int(b[i][j][0]+j*32/scale+16/scale)
					y = int(b[i][j][1]+i*32/scale+16/scale)
					w = int(b[i][j][2])
					h = int(b[i][j][3])
					# print(b[i][j])
					# cv2.circle(img,(x,y),5,(0,0,255),-1)
					cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,255,0),1)

	#cv2.imshow('img',img)
	#cv2.waitKey(wait)
	# cv2.destroyAllWindows()


def train_RPN(imgholder,biasholder,confholder,biaslosses,conflosses,train_s,RPNcb):
	MAXITER = 100000000
	BSIZE = 32

	#config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.2
	#sess = tf.Session(config=tf.ConfigProto( allow_soft_placement=True, log_device_placement=True))
	with tf.Session() as sess:
		M.loadSess('drive/Colab/Multi_loss/training/model_RPN/',sess,init=True)
		saver = tf.train.Saver()
		reader = data_reader('drive/Colab/Multi_loss/training/annotationC.txt')
		print('Reading finish')
		for iteration in range(MAXITER):
			train_batch,nscale = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch0 = [i[1] for i in train_batch]
			conf_batch0 = [i[2] for i in train_batch]
			bias_batch2 = [i[3] for i in train_batch]
			conf_batch2 = [i[4] for i in train_batch]

			index= int(1-nscale)	#bias0[0],bias1[0],bias2[0],bias3[0],conf0[0],conf1[0],conf2[0],conf3[0]
			scale = 4**float(nscale)
			if index == 0:
				bias_batch = bias_batch0
				conf_batch = conf_batch0
			else:
				bias_batch = bias_batch2
				conf_batch = conf_batch2

			img_batch = np.float32(img_batch)
			feeddict = {imgholder:img_batch, biasholder:bias_batch, confholder:conf_batch}
			loss_b,loss_c,_,b,c= sess.run([biaslosses[index],conflosses[index],train_s[index],RPNcb[index],RPNcb[(int(index+2))]],feed_dict=feeddict)
			if iteration%10==0:
				log_loss_b = np.log(loss_b)
				log_loss_c = np.log(loss_c)
				print('Iter:',iteration,'\tIndex:',index,'\tloss_b:',loss_b,'\tloss_c:',loss_c,'\tlog_b:',log_loss_b,'\tlog_c:',log_loss_c)
				#print(b,c)
				#img = img_batch[0].astype(np.uint8)
				#draw(img,c[0],b[0],scale,wait=5)
				#img=cv2.resize(img,(960,540))
				#cv2.imshow('img',img)
				#cv2.waitKey(100)	
				#draw(img,conf_batch[0],bias_batch[0],wait=5)
			if iteration%3000==0 and iteration!=0:
				saver.save(sess,'drive/Colab/Multi_loss/training/model_RPN/'+str(iteration)+'.ckpt')

def train_veri(imgholder,RPNcb,croppedholder,veri_conf_holder,veri_conf_loss,veri_train_step,veri_conf,veri_accuracy):
	MAXITER = 5000000000
#	MAXITER = 1
	BSIZE = 32
	with tf.Session() as sess:
		M.loadSess('drive/Colab/Multi_loss/training/model_VERI/',sess,init=True)
		M.loadSess(modpath='drive/Colab/Multi_loss/training/model_RPN/12000.ckpt',sess=sess,var_list=M.get_trainable_vars('mainModel'))
		saver = tf.train.Saver()
		reader = data_reader('drive/Colab/Multi_loss/training/annotationC.txt')
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
			if iteration%500==0 and iteration!=0:
				saver.save(sess,'drive/Colab/Multi_loss/training/model_VERI/'+str(iteration)+'.ckpt')



create_graph=graph.build_graph()
RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb = create_graph.graphs()
#croppedholder, veri_conf_holder, veri_conf_loss, veri_train_step, veri_conf, veri_accuracy = build_graph2()

## Train RPN

#train_RPN(RPNholders[0],RPNholders[1],RPNholders[2], RPNlosses[0],RPNlosses[1], train_steps[0],RPNcb)

## Train verification

#train_veri(RPNholders[0],RPNcb[0],RPNcb[1],croppedholder, veri_conf_holder, veri_conf_loss, veri_train_step, veri_conf, veri_accuracy)
train_veri(RPNholders[0],RPNcb,veriholders[0],veriholders[1],verilosses[0],train_steps[1],vericb[0],vericb[1])
