import numpy as np 
import datareader2
import cv2
import netpart
import net_veri
import tensorflow as tf 
import model as M 

import os
if not os.path.exists('./modelveri/'):
	os.mkdir('./modelveri/')

threshold = 0.4
MAXITER = 100000

def get_img_coord(img,c,b,multip):
	res = []
	c = c[0]
	b = b[0]
	row,col,_ = b.shape
	c = c.reshape([-1])
	ind = c.argsort()[-5:][::-1]
	for aaa in ind:
		i = aaa//col
		j = aaa%col 
		x = int(b[i][j][0])+j*multip+multip//2
		y = int(b[i][j][1])+i*multip+multip//2
		w = int(b[i][j][2])
		h = int(b[i][j][3])
		try:
			M = np.float32([[1,0,-(x-int(w*1.5)//2)],[0,1,-(y-int(h*1.5)//2)]])
			cropped = cv2.warpAffine(img,M,(int(w*1.5),int(h*1.5)))
			cropped = cv2.resize(cropped,(32,32))
			res.append([cropped,[x,y,w,h]])
		except:
			continue
	return res 


def get_iou(inp1,inp2):
	x1,y1,w1,h1 = inp1[0],inp1[1],inp1[2],inp1[3]
	x2,y2,w2,h2 = inp2[0],inp2[1],inp2[2],inp2[3]
	xo = min(abs(x1+w1/2-x2+w2/2), abs(x1-w1/2-x2-w2/2))
	yo = min(abs(y1+h1/2-y2+h2/2), abs(y1-h1/2-y2-h2/2))
	if abs(x1-x2) > (w1+w2)/2 or abs(y1-y2) > (h1+h2)/2:
		return 0
	if abs(float((x1-x2)*2)) < abs(w1-w2):
		xo = min(w1, w2)
	if abs(float((y1-y2)*2)) < abs(h1-h2):
		yo = min(h1, h2)
	overlap = xo*yo
	total = w1*h1+w2*h2-overlap
	return float(overlap)/total

def get_lb(p1,p2):
	ioumax = 0.
	for p in p2:
		iou = get_iou(p1,p)
		# print(p1,p)
		# print(iou)
		if iou>ioumax:
			ioumax = iou
	if ioumax>threshold:
		return 1
	else:
		return 0

def crop(img,bs,cs,coord):
	multi = [8,32,128]
	res = []
	lbs = []
	crds = []
	for i in range(3):
		buff = get_img_coord(img,cs[i],bs[i],multi[i])
		res += buff
	for c in coord:
		x,y,w,h = c
		x,y,w,h = int(x),int(y),int(w),int(h)
		crds.append([x-w//2,y-h//2,w,h])
	for item in res:
		cropped = item[0]
		coord = item[1]
		lb = get_lb(coord,crds)
		lbs.append([cropped,lb])
		# x,y,w,h = 
		# cv2.rectangle(img,())
	return lbs

reader = datareader2.reader(scale_range=[0.05,1.2])
b0,b1,b2,c0,c1,c2 = netpart.model_out

with tf.Session() as sess:
	saver = tf.train.Saver()
	M.loadSess('./model/',sess,init=True,var_list=M.get_trainable_vars('MSRPN'))
	for i in range(MAXITER):
		img,coord = reader.get_img()
		buff_out = sess.run([b0,b1,b2,c0,c1,c2],feed_dict={netpart.inpholder:[img]})
		bs,cs = buff_out[:3],buff_out[3:]
		lbs = crop(img,bs,cs,coord)
		train_imgs = [k[0] for k in lbs]
		train_labs = [k[1] for k in lbs]
		# for item in lbs:
		# 	cv2.imshow('ad',item[0])
		# 	print(item[1])
		# 	cv2.waitKey(0)
		ls,ac,_ = sess.run([net_veri.loss,net_veri.accuracy,net_veri.ts],
			feed_dict={net_veri.inputholder:train_imgs,net_veri.labelholder:train_labs})
		if i%10==0:
			print('Iter:%d\tLoss:%.4f\tAcc:%.4f'%(i,ls,ac))
		if i%3000==0 and i>0:
			saver.save(sess,'./modelveri/%d.ckpt'%i)
