import tensorflow as tf
import graph
import data_reader as data_reader
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
	MAXITER = 1000000000

	#config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.2
	#sess = tf.Session(config=tf.ConfigProto( allow_soft_placement=True, log_device_placement=True))
	with tf.Session() as sess:
		M.loadSess('drive/Colab/Multi_loss/training/model_RPN/',sess,init=True)
		saver = tf.train.Saver()
		reader = data_reader.reader('drive/Colab/Multi_loss/training/annotationC.txt',height=270*2,width=480*2,scale_range=[0.02,1.3],lower_bound=2,upper_bound=4,index_multiplier=1)
		print('Reading finish')
		for iteration in range(MAXITER):
			img, train_dic = reader.get_img()
			for k in train_dic:
				feeddict = {imgholder:[img],biasholder:[train_dic[k][1]],confholder:[train_dic[k][0]]}
				loss_b,loss_c,_,b,c= sess.run([biaslosses[k],conflosses[k],train_s[k],RPNcb[k],RPNcb[(int(k+3))]],feed_dict=feeddict)

			if iteration%200==0:
				log_loss_b = np.log(loss_b)
				log_loss_c = np.log(loss_c)
				print('Iter:',iteration,'\tIndex:',k,'\tloss_b:',loss_b,'\tloss_c:',loss_c,'\tlog_b:',log_loss_b,'\tlog_c:',log_loss_c)
				#print(b,c)
				#img = img_batch[0].astype(np.uint8)
				#draw(img,c[0],b[0],scale,wait=5)
				#img=cv2.resize(img,(960,540))
				#cv2.imshow('img',img)
				#cv2.waitKey(100)	
				#draw(img,conf_batch[0],bias_batch[0],wait=5)
			if iteration%50000==0 and iteration!=0:
				saver.save(sess,'drive/Colab/Multi_loss/training/model_RPN/'+str(iteration)+'.ckpt')

def train_veri(imgholder,RPNcb,croppedholder,veri_conf_holder,veri_conf_loss,veri_train_step,veri_conf,veri_accuracy):
	MAXITER = 10000000000
	with tf.Session() as sess:
		M.loadSess('drive/Colab/Multi_loss/training/model_VERI/',sess,init=True)
		M.loadSess(modpath='drive/Colab/Multi_loss/training/model_RPN/150000.ckpt',sess=sess,var_list=M.get_trainable_vars('mainModel'))
		saver = tf.train.Saver()
		reader = data_reader.reader('drive/Colab/Multi_loss/training/annotationC.txt',height=270*2,width=480*2,scale_range=[0.02,1.3],lower_bound=2,upper_bound=4,index_multiplier=1)
		print('Reading finish')
		for iteration in range(MAXITER):
			img, train_dic = reader.get_img()
			for k in train_dic:
				feeddict = {imgholder:[img]}
				b,c= sess.run([RPNcb[k],RPNcb[(int(k+3))]],feed_dict=feeddict)

			if k==0:
				multip = 8
			elif k==1:
				multip = 16
			else:
				multip = 32

			cropped_batch = []
			veri_conf_batch = []
			croppedImages, labels = F.crop_original(img, train_dic[k][1], train_dic[k][0],b[0],c[0],multip)

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
			if iteration%500==0:
				print('Iter:',iteration,'\tIndex:',k,'\tLoss:',loss,'\taccuracy:',acc)		
			if iteration%50000==0 and iteration!=0:
				saver.save(sess,'drive/Colab/Multi_loss/training/model_VERI/'+str(iteration)+'.ckpt')



create_graph=graph.build_graph()
RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb = create_graph.graphs()
#croppedholder, veri_conf_holder, veri_conf_loss, veri_train_step, veri_conf, veri_accuracy = build_graph2()

## Train RPN

train_RPN(RPNholders[0],RPNholders[1],RPNholders[2], RPNlosses[0],RPNlosses[1], train_steps[0],RPNcb)

## Train verification

#train_veri(RPNholders[0],RPNcb[0],RPNcb[1],croppedholder, veri_conf_holder, veri_conf_loss, veri_train_step, veri_conf, veri_accuracy)
#train_veri(RPNholders[0],RPNcb,veriholders[0],veriholders[1],verilosses[0],train_steps[1],vericb[0],vericb[1])
