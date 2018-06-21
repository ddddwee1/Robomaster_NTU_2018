import rune_recog_template_gw
import conv
import cv2
import time
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS,30)
#fps = cap.get(cv2.CAP_PROP_FPS)
#print 'fps',fps
WHITE = (255,255,255)
BLACK = (0,0,0)

image_width = 1024
image_height = 768 

cap.set(3, image_width);
cap.set(4, image_height);

while True:
	leftRect = []
	rightRect = []
	left_rect = []
	right_rect = []
	row_left_rect = []
	row_right_rect = []
	#img = cv2.imread('b.jpg',3)
	_,image = cap.read()
	cv2.imshow('',image)

	#print img
	#img1=np.array(img)
	#print img1.shape

	# graycale, blurring it, and computing an edge map
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#blurred = cv2.GaussianBlur(gray, (5,5), 0)
	blurred = cv2.bilateralFilter(gray, 3, 139, 139)
	#ret, th_img = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
	#cv2.imshow('cccc',blurred)
	edged = cv2.Canny(blurred, 120, 240,L2gradient=True)
	#ret, th_img = cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.waitKey(1)
	_, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:

		if cv2.contourArea(contour)<500 or cv2.contourArea(contour)>1500:
			continue

		x,y,w,h = cv2.boundingRect(contour)

		if (w / h) < 1.0 or (w / h) > 2.5:
			continue

		cv2.drawContours(edged, [contour], -1, (255, 255, 255), 3)

		if x < (image_width//2):
			leftRect.append(contour)
		else:
			rightRect.append(contour)

	if len(leftRect)< 5 or len(rightRect)< 5 :
		continue

	find_left = True

	for i in range(len(leftRect)):
		xi,_,_,_ = cv2.boundingRect(leftRect[i])
		for j in range(len(leftRect)):
			j1=4-j
			xj1,_,_,_ = cv2.boundingRect(leftRect[j1])
			if abs(xi - xj1) > 50:
				find_left = False

		if find_left == True:
			left_rect.append(leftRect[i])

		find_left = True

	for i in range(len(left_rect)):
		x1,y1,w1,h1 = cv2.boundingRect(left_rect[i])
		row_left_rect.append(y1)
		cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)

	find_right = True

	for i in range(len(rightRect)):
		xi,_,_,_ = cv2.boundingRect(rightRect[i])
		for j in range(len(rightRect)):
			j1=4-j
			xj1,_,_,_ = cv2.boundingRect(rightRect[j1])
			if abs(xi - xj1) > 50:
				find_right = False

		if find_right == True:
			right_rect.append(rightRect[i])

		find_right = True

	for i in range(len(right_rect)):
		x2,y2,w2,h2 = cv2.boundingRect(right_rect[i])
		row_right_rect.append(y2)
		cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)

	if len(row_left_rect) == 0 or len(row_right_rect) == 0:
		continue

	#print row_left_rect

	tl_index = np.argmin(row_left_rect)
	bl_index = np.argmax(row_left_rect)
	tr_index = np.argmin(row_right_rect)
	br_index = np.argmax(row_right_rect)

	x1,y1,w1,h1=cv2.boundingRect(left_rect[tl_index])
	x2,y2,w2,h2=cv2.boundingRect(left_rect[bl_index])
	x3,y3,w3,h3=cv2.boundingRect(right_rect[tr_index])
	x4,y4,w4,h4=cv2.boundingRect(right_rect[br_index])

	sr_tl = [x1,y1]
	sr_bl = [x2,y2+h2]
	sr_tr = [x3+w3,y3]
	sr_br = [x4+w4,y4+h4]


	pts1 = np.float32([sr_tl,sr_tr,sr_bl,sr_br])
	pts2 = np.float32([[0, 0],[300, 0],[0, 150],[300, 150]])
#	pts2 = pts2 + np.float32([70, 35])
	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(image,M,(300,150))
	dst = cv2.Canny(dst, 50, 255, 255,L2gradient=True )

	digits_rect = [(51,3),(125,3),(200,3),(51,58),(125,58),(200,58),(51,113),(125,113),(200,113)]
	digit_imgs = []
	abc = 0
	for x,y in digits_rect:
		#cv2.rectangle(dst,(x,y),(x+48,y+34),(255,0,0),2)
		buf = dst[y:y+32,x:x+48]
		#buf = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
		buf = cv2.copyMakeBorder(buf,1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)
		buf = cv2.resize(buf,(28,28))
		cv2.imshow('bb'+abc,buf)
		#cv2.waitKey(0)
		buf = buf.reshape([-1])
		digit_imgs.append(buf)

	cv2.waitKey(0)
	scr = conv.get_pred(digit_imgs)
	print(scr)

	dst1 = cv2.warpPerspective(image,M,(300,150))
	dst1 = cv2.Canny(dst, 50, 255, 255,L2gradient=True )
	digits_rect = [(18, 18), (122, 18), (226, 18), (330, 18), (434, 18)]

	digit_imgs = []
	for x,y in digits_rect:
		#cv2.rectangle(dst,(x,y),(x+48,y+34),(255,0,0),2)
		buf = dst[y:y+32,x:x+48]
		#buf = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
		buf = cv2.copyMakeBorder(buf,1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)
		buf = cv2.resize(buf,(28,28))
		cv2.imshow('bb'+abc,buf)
		#cv2.waitKey(0)
		buf = buf.reshape([-1])
		digit_imgs.append(buf)

	scr_7seg = conv.get_pred(digit_imgs)
	print(scr_7seg)

	scr_7seg = conv.get_pred(digit_imgs)
	cv2.imshow('a',dst)
	cv2.imshow('b',edged)
#	cv2.imshow('b', dst)
	cv2.waitKey(1)

