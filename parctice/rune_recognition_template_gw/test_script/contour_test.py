import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,60)
fps = cap.get(cv2.CAP_PROP_FPS)
print 'fps',fps
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

while True:


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

	#cv2.imshow('',th3)

	cv2.waitKey(1)
	_, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		#contour = cv2.convexHull(contour)
		peri = cv2.arcLength(contour,True)
		contour = cv2.approxPolyDP(contour, 0.00001 * peri, True)
		if cv2.contourArea(contour)<500 or cv2.contourArea(contour)>2500:
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
	cv2.imshow('aaaa',edged)
#	if len(leftRect)< 5 or len(rightRect)< 5 :
#		continue


	for i in range(len(leftRect)):
		find_left = False
		xi,_,_,_ = leftRect[i]
		for j in range(len(leftRect)-1):
			j1=1+j
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
		for j in range(len(rightRect)-1):
			j1=1+j
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

	cv2.imshow('a', dst)
	cv2.imshow('b',edged)
#	cv2.imshow('b', dst)
	cv2.waitKey(1)
