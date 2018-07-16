import tensorflow as tf
import graph
from data_reader import data_reader
import model as M
import numpy as np 
import cv2
import Functions as F

def draw(img,b,inds,c_veri,wait=0):
	img = img.copy()
	for k in range(len(c_veri)):
		if c_veri[k]>0:
			ind = inds[k]
			i = ind//16
			j = ind%16
			x = int(b[i][j][0])+j*16+8
			y = int(b[i][j][1])+i*16+8
			w = int(b[i][j][2])
			h = int(b[i][j][3])
			x = x-w
			y = y-h
			cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,255,0),2)
			# print('draw')
	cv2.imshow('RPN+VERI',img)
	cv2.waitKey(wait)
	# cv2.destroyAllWindows()

def draw2(img,c,b,wait=0):
	img= img.copy()
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
	cv2.imshow('RPN',img)
	cv2.waitKey(wait)
	# cv2.destroyAllWindows()

RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb, feature_map = graph.build_graph(False)
def test_veri(imgholder,conf,bias,croppedholder,veri_conf):
	with tf.Session() as sess:
		M.loadSess('./model_VERI/',sess)
		reader = data_reader('merge.txt')
		size = reader.get_size()
		for i in range(size):
			img = reader.get_img(i)
			# img = cv2.imread('Image315.jpg')
			# img = cv2.resize(img,(256,256))
			c,b = sess.run([conf,bias],feed_dict={imgholder:[img]})
			cropped_imgs, inds = F.crop_original_test(img,c[0],b[0])
			c_veri = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs})
			draw2(img,c[0],b[0],1)
			draw(img,b[0],inds,c_veri[0],300)

test_veri(RPNholders[0],RPNcb[0],RPNcb[1],veriholders[0],vericb[0])
