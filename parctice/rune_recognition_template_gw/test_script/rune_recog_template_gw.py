#import rune_recog_template
import conv_deploy
#import conv_7seg
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS,30)
fps = cap.get(cv2.CAP_PROP_FPS)
#print 'fps',fps
WHITE = (255,255,255)
BLACK = (0,0,0)

image_width = 1024
image_height = 768

cap.set(3, image_width);
cap.set(4, image_height);

cap.set(14, 0.0)  #exposure
cap.set(10, 0.05) #brightness


kernel2 = np.ones((2,2),np.uint8)
kernel3 = np.ones((3,3),np.uint8)
kernel4 = np.ones((4,4),np.uint8)
#exp3 = cap.get(10)
#print exp3

while(cap.isOpened()):


	leftRect = []
	rightRect = []
	left_rect = []
	right_rect = []
	row_left_rect = []
	row_right_rect = []
	#img = cv2.imread('b.jpg',3)
	_,image = cap.read()

	#print img
	#img1=np.array(img)
	#print img1.shape

	# graycale, blurring it, and computing an edge map
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(3,3),0)
	blur = cv2.GaussianBlur(blur,(3,3),0)
	blurred = cv2.bilateralFilter(blur, 3, 333, 333)
	_,th3 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	edged = cv2.Canny(blurred, 255, 255)
	#ret, th_img = cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	cv2.imshow('',image)

	cv2.waitKey(1)
	_, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		#contour = cv2.convexHull(contour)
		peri = cv2.arcLength(contour,True)
		contour = cv2.approxPolyDP(contour, 0.00001 * peri, True)
		if cv2.contourArea(contour)<300 or cv2.contourArea(contour)>2500:
			continue

		x,y,w,h = cv2.boundingRect(contour)

		if (w / h) < 1.0 or (w / h) > 2.5:
			continue
		#print cv2.contourArea(contour)
		cv2.drawContours(edged, [contour], -1, (255, 255, 255), 3)

		if x < (image_width//2):
			leftRect.append([x,y,w,h])
		else:
			rightRect.append([x,y,w,h])
	#cv2.imshow('aaaa',edged)
#	if len(leftRect)< 5 or len(rightRect)< 5 :
#		continue


	for i in range(len(leftRect)):
		find_left = False
		xi,_,_,_ = leftRect[i]
		for j in range(len(leftRect)-i-1):
			j1=1+i+j
			xj1,_,_,_ = leftRect[j1]
			if abs(xi - xj1) < 50:
				find_left = True

		if find_left == True:
			left_rect.append(leftRect[i])


	for i in range(len(left_rect)):
		x1,y1,w1,h1 = left_rect[i]
		row_left_rect.append(y1)
		#cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)

	for i in range(len(rightRect)):
		find_right = False
		xi,_,_,_ = rightRect[i]
		for j in range(len(rightRect)-i-1):
			j1=1+i+j
			xj1,_,_,_ = rightRect[j1]
			if abs(xi - xj1) < 50:
				find_right = True

		if find_right == True:
			right_rect.append(rightRect[i])

	for i in range(len(right_rect)):
		x2,y2,w2,h2 = right_rect[i]
		row_right_rect.append(y2)
		#cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)

	if len(row_left_rect) == 0 or len(row_right_rect) == 0:
		continue
	#print row_left_rect

	tl_index = np.argmin(row_left_rect)
	bl_index = np.argmax(row_left_rect)
	tr_index = np.argmin(row_right_rect)
	br_index = np.argmax(row_right_rect)

	x1,y1,w1,h1=left_rect[tl_index]
	x2,y2,w2,h2=left_rect[bl_index]
	x3,y3,w3,h3=right_rect[tr_index]
	x4,y4,w4,h4=right_rect[br_index]

	sr_tl = [x1,y1]
	sr_bl = [x2,y2+h2]
	sr_tr = [x3+w3,y3]
	sr_br = [x4+w4,y4+h4]


	pts1 = np.float32([sr_tl,sr_tr,sr_bl,sr_br])
	pts2 = np.float32([[0, 50],[300, 50],[0, 200],[300, 200]])
#	pts2 = pts2 + np.float32([70, 35])
	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(image,M,(300,200))

	digits_rect = [(51,54),(125,54),(200,54),(51,109),(125,109),(200,109),(51,163),(125,163),(200,163)]
	digit_imgs = []
	abc = 0
	for x,y in digits_rect:
		#cv2.rectangle(dst,(x,y),(x+50,y+34),(0,0,255),1)
		buf =  dst[y:y+32,x:x+50]
		buf = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
		#buf = cv2.equalizeHist(buf)
		#buf = cv2.fastNlMeansDenoisingColored(buf,None,10,10,7,21)
		_,buf = cv2.threshold(buf,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#buf = cv2.adaptiveThreshold(buf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,1)
		#buf = cv2.copyMakeBorder(buf,1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)
		buf = cv2.erode(buf,kernel2,iterations = 1)
		buf = cv2.resize(buf,(24,24))
		buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
		#cv2.imshow('cv',buf)
		#cv2.waitKey(0)
		buf = 255 - buf
		buf = np.float32(buf) / 255.

		buf = buf.reshape([-1])
		digit_imgs.append(buf)

#	handwritten_num_raw = conv_deploy.get_pred(digit_imgs)
#	print(handwritten_num_raw)

#	handwritten_dict = {}
#	num_7seg_dict = {}

#	# filter handwritten_num
#	for i in range(len(handwritten_num_raw)):
#		handwritten_dict[handwritten_num_raw[i]] = np.count_nonzero(handwritten_num_raw[i])

#	if len(handwritten_dict) >= 9 :
#		handwritten_num= handwritten_num_raw
#		#print handwritten_num
#	else:
#		pass

	digits_7seg_rect = [(100, 0), (121, 0), (142, 0), (162, 0), (183, 0)]

	digit_7seg_imgs = []

	for x,y in digits_7seg_rect:
		#cv2.rectangle(dst,(x,y),(x+19,y+33),(255,0,0),1)
		img_sevseg = dst[y:y+32,x:x+19]

		#buf = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
		img_sevseg=cv2.cvtColor(img_sevseg, cv2.COLOR_BGR2HSV)
		img_sevseg_red = img_sevseg[:,:,2].copy()
		#img_sevseg_red = cv2.inRange(img_sevseg_red, 210, 255)
		img_sevseg_red = cv2.morphologyEx(img_sevseg_red, cv2.MORPH_OPEN, kernel2)
		_,buf = cv2.threshold(img_sevseg_red,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#buf=cv2.dilate(buf,kernel,iterations = 1)
		#buf=cv2.erode(buf,kernel,iterations = 2)
		buf = cv2.bitwise_not(buf)
		buf=cv2.dilate(buf,kernel2,iterations = 1)

		buf = cv2.resize(buf,(24,24))
		buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
		#cv2.imshow('bb',buf)
		#cv2.waitKey(0)
		buf = buf.reshape([-1])
		buf = 255 - buf
		buf = np.float32(buf) / 255.
		buf = buf.reshape([-1])
		digit_7seg_imgs.append(buf)

#	scr_7seg_raw = conv_deploy.get_pred_7seg(digit_7seg_imgs)
#	print (scr_7seg_raw)
	#print scr_7seg_raw
#	for i in range(len(scr_7seg_raw)):
#		num_7seg_dict[scr_7seg_raw[i]] = np.count_nonzero(scr_7seg_raw[i])

#	if len(num_7seg_dict) >= 5 :
#		scr_7seg= scr_7seg_raw
#		#print scr_7seg
#	else:
#		continue
#	print(scr_7seg)

	flamingdigits_rect = [(61,54),(135,54),(210,54),(61,109),(135,109),(210,109),(61,163),(135,163),(210,163)]
	digit_imgs = []
	abc = 0
	for x,y in flamingdigits_rect:
		#cv2.rectangle(dst,(x,y),(x+50,y+34),(0,0,255),1)
		buf =  dst[y:y+32,x:x+30]
		#cv2.imshow('cv',buf)
		img_FD = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
		#buf=cv2.erode(img_FD,kernel2,iterations = 3)
		_,buf = cv2.threshold(img_FD,200,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
		#_, contours, hierarchy = cv2.findContours(buf.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		edged = cv2.Canny(buf, 255, 255)
		_, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(buf, contours, -1, 255,-1)
		buf = cv2.bitwise_not(buf)
		buf = cv2.resize(buf,(24,24))
		buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
		#cv2.imshow('cv',buf)
		#cv2.waitKey(0)
		buf = 255 - buf
		buf = np.float32(buf) / 255.

		buf = buf.reshape([-1])
		digit_imgs.append(buf)


	scr_FD_raw = conv_deploy.get_pred_flaming(digit_imgs)
#	print scr_FD_raw

#	num_FD_dict = {}
#	for i in range(len(scr_FD_raw)):
#		num_FD_dict[scr_FD_raw[i]] = np.count_nonzero(scr_FD_raw[i])

#	if len(num_FD_dict) >= 9 :
#		scr_FD= scr_FD_raw
#		#print scr_7seg
#	else:
#		scr_FD = False
#		pass
#	print(scr_FD)

	cv2.imshow('a', dst)
	#cv2.imshow('b',edged)
#	cv2.imshow('b', dst)
	cv2.waitKey(1)

