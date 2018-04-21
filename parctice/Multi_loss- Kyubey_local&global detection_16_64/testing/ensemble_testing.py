import tensorflow as tf
import graph
from data_reader1 import data_reader
import model as M
import numpy as np 
import cv2
import Functions as F

KNOWN_DISTANCE = 2.0
KNOWN_WIDTH = 40.0

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

def draw(img,b,c,c_veri,scale,name,wait=0):
	img = img.copy()
	row,col,_ = b.shape
	c = c.reshape([-1])
	indices = c.argsort()[-5:][::-1]      #we get the indices of flattened array in descending order
	for k in range(len(c_veri)):
		if c_veri[k]==c_veri.max() and c_veri[k]>-3.0:
			ind = indices[k]
			i = int(np.floor(ind/(col)))
			j = int(np.floor(ind%(col)))
			x = int((b[i][j][0]+j*64/scale+32/scale))
			y = int((b[i][j][1]+i*64/scale+32/scale))
			w = int(b[i][j][2])
			h = int(b[i][j][3])
			#print (x,y,w,h,c_veri.max())
			return (x,y,w,h,c_veri.max())
			# print('draw')
	return(0,0,0,0,0)
	# cv2.destroyAllWindows()

def draw2(img,c,b,scale,name,wait=0):
	# print(c.shape)
	# print(b.shape)
	# print(c.max())
	c5 = c
	c5 = c5.reshape([-1])
	ind = c5.argsort()[-5:][::-1]
	row,col,_ = b.shape
	for i in range(row):
		for j in range(col):
			if c[i][j][0] == c.max():
				x = int((b[i][j][0]+j*64/scale+32/scale))
				y = int((b[i][j][1]+i*64/scale+32/scale))
				w = int(b[i][j][2])
				h = int(b[i][j][3])
				#print (scale , c.max())
				# print(b[i][j])
				# cv2.circle(img,(x,y),5,(0,0,255),-1)
				cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(0,255,0),4)

			else:
				if c[i][j][0]>-2.0:
					x = int((b[i][j][0]+j*64/scale+32/scale))
					y = int((b[i][j][1]+i*64/scale+32/scale))
					w = int(b[i][j][2])
					h = int(b[i][j][3])
					# print(b[i][j])
					# cv2.circle(img,(x,y),5,(0,0,255),-1)
					cv2.rectangle(img,(x-w,y-h),(x+w,y+h),(255,0,0),2)
	img = cv2.resize(img,(480,270))
	cv2.imshow(name,img)
	cv2.waitKey(wait)

create_graph=graph.build_graph()
RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb = create_graph.graphs(False)

def test_veri(imgholder,RPNcb,croppedholder,veri_conf):
	with tf.Session() as sess:
		M.loadSess('./model_VERI/',sess)
		reader = data_reader('2018-03-17-164409.webm')
		size = reader.get_size()
		state =0
		for i in range(size):
			if state == 0: #global
				img = reader.get_img(i)
				img1 = np.float32(img).copy()
				img1 = np.uint8(img1)
				img2 = np.float32(img).copy()
				img2 = np.uint8(img2)
				img3 = np.float32(img).copy()
				img3 = np.uint8(img3)
				img4 = np.float32(img).copy()
				img4 = np.uint8(img4)
				#img = cv2.imread('Image315.jpg')
				#print (img)
				#img = cv2.resize(img,(256,256))
				b0,b2,c0,c2= sess.run([RPNcb[0],RPNcb[1],RPNcb[2],RPNcb[3]],feed_dict={imgholder:[img]})	# getting c,b output from the RPN
	#			print (c0.max() , c2.max())
	#			if c0.max() > c2.max():
	#				scale = 4.0
	#				b=b0
	#				c=c0
	#			else:
	#				scale = 1.0
	#				b=b2
	#				c=c2

				cropped_imgs0,inds0 = F.crop_original_test(img,b0[0],c0[0],4)
				c_veri0 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs0})

				cropped_imgs2,inds2 = F.crop_original_test(img,b2[0],c2[0],1)
				c_veri2 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs2})

				#draw2(img1,c0[0],b0[0],4,name='RPN_0',wait=1)
				#draw2(img2,c2[0],b2[0],1,name='RPN_2',wait=1)
				x0,y0,w0,h0,c0 = draw(img3,b0[0],c0[0],c_veri0,4,name='VERI_0',wait=1)
				cv2.rectangle(img3,(x0-w0,y0-h0),(x0+w0,y0+h0),(0,255,0),4)
				img3 = cv2.resize(img3,(480,270))
				cv2.imshow('VERI_0',img3)
				cv2.waitKey(1)

				x2,y2,w2,h2,c2 = draw(img4,b2[0],c2[0],c_veri2,1,name='VERI_2',wait=1)
				cv2.rectangle(img4,(x2-w2,y2-h2),(x2+w2,y2+h2),(0,255,0),4)
				img4 = cv2.resize(img4,(480,270))
				cv2.imshow('VERI_2',img4)
				cv2.waitKey(1)

				if c0 > c2:
					nx = x0
					ny = y0
				else:
					nx = x2
					ny = y2

				print ('gobal')
				if x0 > 0 or x2 >0:
					state = 1 

			if state == 1: #local detection
				img = reader.get_img(i)
				img3 = np.float32(img).copy()
				img3 = np.uint8(img3)
				img4 = np.float32(img).copy()
				img4 = np.uint8(img4)
				M1 = np.float32([[1,0,-nx+128],[0,1,-ny+128]])
				img_result = np.float32(img).copy()
				img_result=np.uint8(img_result)
				crop_img = cv2.warpAffine(img_result,M1,(256,256))
#				cv2.imshow('VERI_5',crop_img)
#				cv2.waitKey(0)
				b0,b2,c0,c2= sess.run([RPNcb[0],RPNcb[1],RPNcb[2],RPNcb[3]],feed_dict={imgholder:[crop_img]})

				cropped_imgs0,inds0 = F.crop_original_test_c(crop_img,b0[0],c0[0],4)
				c_veri0 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs0})

				cropped_imgs2,inds2 = F.crop_original_test_c(crop_img,b2[0],c2[0],1)
				c_veri2 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs2})


				x0c,y0c,w0c,h0c,c0c = draw(crop_img,b0[0],c0[0],c_veri0,4,name='VERI_0',wait=1)
				if x0c < 1 and y0c < 1:
					state = 0 
					continue
				x0 = x0c + nx -128
				y0 = y0c + ny -128

				if x0 < 0 or y0 < 0:
					state = 0 
					continue

				cv2.rectangle(img3,(x0-w0c,y0-h0c),(x0+w0c,y0+h0c),(0,255,0),4)
				img3 = cv2.resize(img3,(480,270))
				cv2.imshow('VERI_0',img3)
				cv2.waitKey(1)

				x2c,y2c,w2c,h2c,c2c = draw(crop_img,b2[0],c2[0],c_veri2,1,name='VERI_2',wait=1)
				if x2c < 1 and y2c < 1:
					state = 0 
					continue
				x2 = x2c + nx -128
				y2 = y2c + ny -128

				if x2 < 0 or y2 < 0:
					state = 0 
					continue

				cv2.rectangle(img4,(x2-w2c,y2-h2c),(x2+w2c,y2+h2c),(0,255,0),4)
				img4 = cv2.resize(img4,(480,270))
				cv2.imshow('VERI_2',img4)
				cv2.waitKey(1)

				if c0c > c2c:
					nx = x0
					ny = y0
					nw = w0c
					nh = h0c

				else:
					nx = x2
					ny = y2
					nw = w2c
					nh = h2c

				depth=distance_to_camera(KNOWN_DISTANCE,KNOWN_WIDTH,nw)
				print ('local','x:',nx,'y:',ny,'w:',nw,'h:',nh,'d',depth,'Veri_0_c:',c0c,'Veri_2_c:',c2c)


test_veri(RPNholders[0],RPNcb,veriholders[0],vericb[0])
