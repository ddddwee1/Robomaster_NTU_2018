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

def draw(img,b,c,c_veri,multip,name,wait=0):
	img = img.copy()
	row,col,_ = b.shape
	c = c.reshape([-1])
	indices = c.argsort()[-5:][::-1]      #we get the indices of flattened array in descending order
	for k in range(len(c_veri)):
		if c_veri[k]>-10.0:
			ind = indices[k]
			i = int(np.floor(ind//(col)))
			j = int(np.floor(ind%(col)))
			x = int((b[i][j][0]+j*multip+multip//2))
			y = int((b[i][j][1]+i*multip+multip//2))
			w = int(b[i][j][2]//2)
			h = int(b[i][j][3]//2)
			#print (x,y,w,h,c_veri.max())
			return (x,y,w,h,c_veri.max())
			# print('draw')
	return(0,0,0,0,0)
	# cv2.destroyAllWindows()


create_graph=graph.build_graph()
RPNholders, veriholders, RPNlosses, verilosses, train_steps, RPNcb, vericb = create_graph.graphs()

def test_veri(imgholder,RPNcb,croppedholder,veri_conf):
	with tf.Session() as sess:
		state = -1
		M.loadSess('./model_VERI/',sess)
		reader = data_reader('2018-03-17-164409.webm')
		size = reader.get_size()
		print('Reading finish')
		for i in range(size):
			if state == -1:#global
				img = reader.get_img(i)
				img0 = np.float32(img).copy()
				img0 = np.uint8(img0)
				img1 = np.float32(img).copy()
				img1 = np.uint8(img1)
				img2 = np.float32(img).copy()
				img2 = np.uint8(img2)

				b0,b1,b2,c0,c1,c2= sess.run([RPNcb[0],RPNcb[1],RPNcb[2],RPNcb[3],RPNcb[4],RPNcb[5]],feed_dict={imgholder:[img]})	# getting c,b output from the RPN

#				if c0.max()>-2.0:
#					cropped_imgs0,inds0 = F.crop_original_test(img0,b0[0],c0[0],8)
#					c_veri0 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs0})

#				if c1.max()>-2.0:
#					cropped_imgs1,inds1 = F.crop_original_test(img1,b1[0],c1[0],16)
#					c_veri1 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs1})

#				if c2.max()>-2.0:
#					cropped_imgs2,inds2 = F.crop_original_test(img2,b2[0],c2[0],32)
#					c_veri2 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs2})


#				x0,y0,w0,h0,c0 = draw(img0,b0[0],c0[0],c_veri0,8,name='VERI_0',wait=1)
#				cv2.rectangle(img0,(x0-w0,y0-h0),(x0+w0,y0+h0),(0,255,0),4)
#				img0 = cv2.resize(img0,(960,540))
#				cv2.imshow('VERI_0',img0)
#				#cv2.waitKey(0)

#				x1,y1,w1,h1,c1 = draw(img1,b1[0],c1[0],c_veri1,16,name='VERI_1',wait=1)
#				cv2.rectangle(img1,(x1-w1,y1-h1),(x1+w1,y1+h1),(0,255,0),4)
#				img1 = cv2.resize(img1,(960,540))
#				cv2.imshow('VERI_1',img1)
#				#cv2.waitKey(0)

#				x2,y2,w2,h2,c2 = draw(img2,b2[0],c2[0],c_veri2,32,name='VERI_2',wait=1)
#				cv2.rectangle(img2,(x2-w2,y2-h2),(x2+w2,y2+h2),(0,255,0),4)
#				img2 = cv2.resize(img2,(960,540))
#				cv2.imshow('VERI_2',img2)
#				cv2.waitKey(0)
				#print (c_veri0.max(),c_veri1.max(),c_veri2.max())
#				if c0.max() > 0 or c1.max() > 0 or c2.max() > 0:
				if c0.max() > -1.0 or c1.max() > -1.0 or c2.max() > -1.0:
					print ('gobal-succeed')
					if c0.max() > c1.max() and c0.max() > c2.max() :
						cropped_imgs0,inds0 = F.crop_original_test(img0,b0[0],c0[0],8)
						c_veri0 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs0})
						x0,y0,w0,h0,c0 = draw(img0,b0[0],c0[0],c_veri0,8,name='VERI',wait=1)
						if x0 <1:
							continue
						print ('gobal-succeed',0,'x:',x0,'y:',y0,'w:',w0,'h:',h0)
						cv2.rectangle(img0,(x0-w0,y0-h0),(x0+w0,y0+h0),(0,255,0),2)
						img0 = cv2.resize(img0,(960,540))
						cv2.imshow('VERI',img0)
						cv2.waitKey(1)
						nx = x0
						ny = y0
						nw = w0
						nh = h0
						state = 0

					if c1.max() > c0.max() and c1.max() > c2.max() :
						cropped_imgs1,inds1 = F.crop_original_test(img1,b1[0],c1[0],16)
						c_veri1 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs1})
						x1,y1,w1,h1,c1 = draw(img1,b1[0],c1[0],c_veri1,16,name='VERI',wait=1)
						if x1 <1:
							continue
						print ('gobal-succeed',1,'x:',x1,'y:',y1,'w:',w1,'h:',h1)
						cv2.rectangle(img1,(x1-w1,y1-h1),(x1+w1,y1+h1),(0,255,0),2)
						img1 = cv2.resize(img1,(960,540))
						cv2.imshow('VERI',img1)
						cv2.waitKey(1)
						nx = x1
						ny = y1
						nw = w1
						nh = h1
						state = 1

					if c2.max() > c0.max() and c2.max() > c1.max() :
						cropped_imgs2,inds2 = F.crop_original_test(img2,b2[0],c2[0],32)
						c_veri2 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs2})
						x2,y2,w2,h2,c2 = draw(img2,b2[0],c2[0],c_veri2,32,name='VERI',wait=1)
						if x2 <1:
							continue
						print ('gobal-succeed',2,'x:',x2,'y:',y2,'w:',w2,'h:',h2)
						cv2.rectangle(img2,(x2-w2,y2-h2),(x2+w2,y2+h2),(0,255,0),2)
						img2 = cv2.resize(img2,(960,540))
						cv2.imshow('VERI',img2)
						cv2.waitKey(1)
						nx = x2
						ny = y2
						nw = w2
						nh = h2
						state = 2


				else:
					img = cv2.resize(img,(960,540))
					cv2.imshow('VERI',img)
					cv2.waitKey(1)
					state = -1
					print ('gobal-fail',c0.max(),c1.max(),c2.max())
#				state = -1



			if state == 0: #local detection_0
				img = reader.get_img(i)
				img3 = np.float32(img).copy()
				img3 = np.uint8(img3)
				M1 = np.float32([[1,0,-nx+48],[0,1,-ny+48]])
				img_result = np.float32(img).copy()
				img_result=np.uint8(img_result)
				crop_img0 = cv2.warpAffine(img_result,M1,(96,96))
#				cv2.imshow('VERI_5',crop_img0)
#				cv2.waitKey(0)
				b0,c0= sess.run([RPNcb[0],RPNcb[3]],feed_dict={imgholder:[crop_img0]})

				cropped_imgs0,inds0 = F.crop_original_test(crop_img0,b0[0],c0[0],8)
				c_veri0 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs0})


				if c_veri0.max() > -2.0:
					x0c,y0c,w0,h0,c0 = draw(img3,b0[0],c0[0],c_veri0,8,name='VERI',wait=1)
					x0 = x0c + nx -48
					y0 = y0c + ny -48

					if x0 < 0 or x0 >960 or y0 < 0 or x0 >540:
						state = -1
						print ('local-fail')
						continue

					cv2.rectangle(img3,(x0-w0,y0-h0),(x0+w0,y0+h0),(0,255,0),2)
					img3 = cv2.resize(img3,(960,540))
					cv2.imshow('VERI',img3)
					cv2.waitKey(1)
					nx = x0
					ny = y0
					nw = w0
					nh = h0
					state = 0
					depth=distance_to_camera(KNOWN_DISTANCE,KNOWN_WIDTH,nw)
					print ('local-',state,'x:',nx,'y:',ny,'w:',nw,'h:',nh,'d',depth,'Veri_0_c:',c0.max())

				else:
					state = -1
					print ('local-fail','Veri_0_c:',c_veri0.max())

			if state == 1: #local detection_0
				img = reader.get_img(i)
				img4 = np.float32(img).copy()
				img4 = np.uint8(img4)
				M1 = np.float32([[1,0,-nx+96],[0,1,-ny+96]])
				img_result = np.float32(img).copy()
				img_result=np.uint8(img_result)
				crop_img1 = cv2.warpAffine(img_result,M1,(192,192))
#				cv2.imshow('VERI_5',crop_img1)
#				cv2.waitKey(0)
				b1,c1= sess.run([RPNcb[1],RPNcb[4]],feed_dict={imgholder:[crop_img1]})

				cropped_imgs1,inds1 = F.crop_original_test(crop_img1,b1[0],c1[0],16)
				c_veri1 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs1})


				if c_veri1.max() >  -1.0:
					x1c,y1c,w1,h1,c1 = draw(img4,b1[0],c1[0],c_veri1,16,name='VERI',wait=1)
					x1 = x1c + nx -96
					y1 = y1c + ny -96

					if x1 < 0 or x1 >960 or y1 < 0 or x1 >540:
						state = -1
						print ('local-fail')
						continue

					cv2.rectangle(img4,(x1-w1,y1-h1),(x1+w1,y1+h1),(0,255,0),2)
					img4 = cv2.resize(img4,(960,540))
					cv2.imshow('VERI',img4)
					cv2.waitKey(1)
					nx = x1
					ny = y1
					nw = w1
					nh = h1
					state = 1
					depth=distance_to_camera(KNOWN_DISTANCE,KNOWN_WIDTH,nw)
					print ('local-',state,'x:',nx,'y:',ny,'w:',nw,'h:',nh,'d',depth,'Veri_1_c:',c1.max())

				else:
					state = -1
					print ('local-fail','Veri_1_c:',c_veri1.max())

			if state == 2: #local detection_0
				img = reader.get_img(i)
				img5 = np.float32(img).copy()
				img5 = np.uint8(img5)
				M1 = np.float32([[1,0,-nx+192],[0,1,-ny+192]])
				img_result = np.float32(img).copy()
				img_result=np.uint8(img_result)
				crop_img2 = cv2.warpAffine(img_result,M1,(384,384))
#				cv2.imshow('VERI_5',crop_img2)
#				cv2.waitKey(0)
				b2,c2= sess.run([RPNcb[2],RPNcb[5]],feed_dict={imgholder:[crop_img2]})

				cropped_imgs2,inds2 = F.crop_original_test(crop_img2,b2[0],c2[0],32)
				c_veri2 = sess.run(veri_conf,feed_dict={croppedholder:cropped_imgs2})


				if c_veri2.max() > -2.0:
					x2c,y2c,w2,h2,c2 = draw(img5,b2[0],c2[0],c_veri2,32,name='VERI',wait=1)
					x2 = x2c + nx -192
					y2 = y2c + ny -192

					if x2 < 0 or x2 >960 or y2 < 0 or x2 >540:
						state = -1
						print ('local-fail')
						continue

					cv2.rectangle(img5,(x2-w2,y2-h2),(x2+w2,y2+h2),(0,255,0),2)
					img5 = cv2.resize(img5,(960,540))
					cv2.imshow('VERI',img5)
					cv2.waitKey(1)
					nx = x2
					ny = y2
					nw = w2
					nh = h2
					state = 2
					depth=distance_to_camera(KNOWN_DISTANCE,KNOWN_WIDTH,nw)
					print ('local-',state,'x:',nx,'y:',ny,'w:',nw,'h:',nh,'d',depth,'Veri_2_c:',c2.max())

				else:
					state = -1
					print ('local-fail','Veri_2_c:',c_veri2.max())


test_veri(RPNholders[0],RPNcb,veriholders[0],vericb[0])
