import tensorflow as tf
import graph
from data_reader import data_reader
import model as M
import numpy as np
import cv2
import Functions

def draw(img, coor, c, wait=0):
	# print(c.shape)
	# print(b.shape)
	# print(c.max())
	for k in range(5):
		for i in range(16):
			for j in range(16):
				tlx = coor[k][0][0]*16
				tly = coor[k][0][1]*16
				brx = (coor[k][1][0]+1)*16
				bry = (coor[k][1][1]+1)*16
				if c[k][0] > 0:
					# print(b[i][j])
					# cv2.circle(img,(x,y),5,(0,0,255),-1)
					cv2.rectangle(img,(tlx,tly),(brx,bry),(0,255,0),2)
				else:
					cv2.rectangle(img,(tlx,tly),(brx,bry),(255,255,0),1)
	cv2.imshow('img',img)
	cv2.waitKey(wait)

def train_verification(imgholder, croppedholder, veri_conf_holder, veri_conf_loss, verfication_train_step, conf, bias, veri_conf, feature_map):
	MAXITER = 50000*2
	BSIZE = 5
	with tf.Session() as sess:
		M.loadSess('',sess,init=True)
		M.loadSess('./model/',sess,init=True,var_list=M.get_trainable_vars('mainModel'))
		saver = tf.train.Saver()
		reader = data_reader('train_list.txt')
		print('Reading finish')
		accData = np.empty((25,1), dtype=np.uint8)
		for iteration in range(MAXITER):
			# Getting the data from data_reader
			train_batch = reader.next_train_batch(BSIZE)
			img_batch = [i[0] for i in train_batch]
			bias_batch = [i[1] for i in train_batch]
			img_batch = np.float32(img_batch)

			# Running the RPN with the original images
			feeddict = {imgholder:img_batch}
			c, b, f = sess.run([conf, bias, feature_map], feed_dict=feeddict)
			# Cropped the feature maps and get the confidence label
			croppedFMs, croppedCoor, labelConf = Functions.crop_featureMaps(f, b, c, bias_batch, BSIZE)

			# Train the Verification Network
			feeddict_veri = {croppedholder:croppedFMs, veri_conf_holder:labelConf}
			vc_loss, _, vc = sess.run([veri_conf_loss, verfication_train_step, veri_conf], feed_dict=feeddict_veri)
			accData = np.append(accData, 1 - abs((vc > 0) - labelConf))
			if iteration == 0:
				accData = np.delete(accData, np.s_[:25])
			if accData.shape[0] > 10000:
				accData = np.delete(accData, np.s_[:25])
			if iteration%10==0:
				print('Iter: ', iteration, '\tLoss: ', vc_loss, '\tAcc: ', np.mean(accData))
				img = img_batch[0].astype(np.uint8)
				draw(img, croppedCoor[:5], labelConf[:5], wait=5)

			if iteration%5000==0 and iteration!=0:
				saver.save(sess,'./model/'+str(iteration)+'.ckpt')

RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, veri_confidence, feature_map = graph.build_graph()
train_verification(RPNholders[0], veriholders[0], veriholders[1], verilosses[0], train_steps[1], RPNcb[0], RPNcb[1], veri_confidence, feature_map)
