import numpy as np
import cv2
import conv
import math
import time

start_time = time.time()
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255,255,255)
EPSILON = 0.00001

# main process
img_ori = cv2.imread('c.jpg')
#img = img_ori.copy()
# Enlarge the image, Optional
img = cv2.resize(img_ori, (1300,1300), interpolation = cv2.INTER_LINEAR)
# Convert to grayscale
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur the image to remove noises
img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
# Using otsu's binarization to threshold the image and find the scoreboard rectangle
ret, th_img = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Find the contours from the thresholded image
_, cnts, hier = cv2.findContours(th_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Finding the scoreboard rectangles
side_rects = []
rect_rects = []
upright_rects = []
rotated_rects = []
test_rects = []

hier = hier[0]
center = 0

# 1. Picking the contour which does not contain another contour
# 2. Is a rectangle in shape

"""
for cnt in cnts:
	x,y,w,h = cv2.boundingRect(cnt)
	side_rects.append([x,y,w,h])
"""
for idx in range(len(hier)):
	if hier[idx][2] == -1:
		cnt = cnts[idx]
		cnt_area = cv2.contourArea(cnt)
		x,y,w,h = cv2.boundingRect(cnt)
		#side_rects.append([x,y,w,h])
		rr = cv2.minAreaRect(cnt)
		rect_area = w*h
		rr_area = rr[1][0]*rr[1][1]
		if rr_area > EPSILON:
			extent = float(cnt_area) / rect_area
			if extent > 0.8:
			#print(rr_area)
				side_rects.append([x,y,w,h])
				center += x

# 3. It forms a line - This is done by doing hough line transformation
#img_shape = img.shape()
#max_line_length = math.sqrt(img_shape[0]**2 + img_shape[1]**2)
#max_line_length = int(max_line_length) + 1
#line_param_matrix = np.zeros((60,max_line_length), dtype=int)
#for rect in rect_rects:
#	center = rect[0][0]
#	for theta in range(60):
#		rho = center[0]

center = center / len(side_rects)
rects = side_rects
sr_tl = [0,2000]
sr_bl = [0,0]
sr_tr = [0,2000]
sr_br = [0,0]
for rect in side_rects:
    if rect[0] < center:
        if rect[1] < sr_tl[1]:
            sr_tl = [rect[0]+rect[2]/2, rect[1]+rect[3]/2]

        if rect[1] > sr_bl[1]:
            sr_bl = [rect[0]+rect[2]/2, rect[1]+rect[3]/2]

    elif rect[0] > center:
        if rect[1] < sr_tr[1]:
            sr_tr = [rect[0]+rect[2]/2, rect[1]+rect[3]/2]

        if rect[1] > sr_br[1]:
            sr_br = [rect[0]+rect[2]/2, rect[1]+rect[3]/2]

pts1 = np.float32([sr_tl,sr_tr,sr_bl,sr_br])
pts2 = np.float32([[50, 280],[1390, 280],[50, 800],[1390, 800]])
pts2 = pts2 + np.float32([70, 35])
M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(1580,1050))
#dst = cv2.fastNlMeansDenoisingColored(dst,None,10,10,7,21)
#dst = cv2.resize(dst,(960,640))

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    pass

# cv2.imshow('aaa',th_img)
# cv2.imshow('bbb',img)
cv2.imshow('ccc', dst)

digits_rect = [(280,275),(650,275),(1020,275),(280,495),(650,495),(1020,495),(280,715),(650,715),(1020,715)]
# recognize digits here
digit_imgs = []
for x,y in digits_rect:
	buf = dst[y:y+160,x:x+280]
	buf = cv2.cvtColor(buf,cv2.COLOR_BGR2GRAY)
	buf = cv2.resize(buf,(24,24))
	buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
	buf = buf.reshape([-1])
	buf = 255 - buf
	buf = np.float32(buf) / 255.
	digit_imgs.append(buf)

scr = conv.get_pred(digit_imgs)
print(scr)

for coord,s in zip(digits_rect,scr):
    x,y = coord
    cv2.rectangle(dst,(x,y),(x+280,y+160),(0,255,0),3)
    cv2.putText(dst,str(s),(x,y+50), FONT, 2,(0,255,0),2,cv2.LINE_AA)

img_sevseg = dst[55:217,518:1062]
#kernel = np.ones((5,5),np.uint8)
#img_sevseg = cv2.erode(img_sevseg,kernel,iterations = 3)
#ret2,img_sevseg = cv2.threshold(img_sevseg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#img_sevseg = cv2.fastNlMeansDenoisingColored(img_sevseg,None,20,40,7,21)
#img_sevseg_hsv = cv2.cvtColor(img_sevseg, cv2.COLOR_BGR2HSV)
#img_sevseg_hsv = img_sevseg_hsv[:,:,2]
img_sevseg_red = img_sevseg[:,:,2].copy()
img_sevseg_float = img_sevseg.copy()
img_sevseg_float = img_sevseg_float.astype(float)
img_sevseg_float = img_sevseg_float / 256
img_sevseg_avg = np.average(img_sevseg, axis = 2, weights=[1./2, 1./4, 1./4])
img_sevseg_avg = img_sevseg_avg.astype(int)
img_sevseg_var = np.var(img_sevseg_float, axis = 2)

#img_sevseg_red = 2*img_sevseg_red / (img_sevseg[:,:,0] + img_sevseg[:,:,1])
#img_sevseg_var = img_sevseg_var.astype(int)
#img_sevseg_red = img_sevseg_red - img_sevseg_avg
#img_sevseg_red = np.absolute(img_sevseg_red)
#img_sevseg_red = img_sevseg_red.astype(float)
#img_sevseg_red = img_sevseg_red / 256
#img_sevseg_red = (img_sevseg_red - img_sevseg_avg)*(img_sevseg_red - img_sevseg_avg) / (4*img_sevseg_var)
#img_sevseg_red[:,:,0] = img_sevseg_red[:,:,0] / (img_sevseg_red[:,:,2]+1)
#img_sevseg_red[:,:,1] = img_sevseg_red[:,:,1] / (img_sevseg_red[:,:,2]+1)
#img_sevseg_red[:,:,2] = img_sevseg_red[:,:,2] / (img_sevseg_red[:,:,2]+1)

img_sevseg_red = cv2.inRange(img_sevseg_red, 210, 255)
kernel = np.ones((7,7),np.uint8)
#img_sevseg_red = cv2.erode(img_sevseg_red,kernel,iterations = 1)
img_sevseg_red = cv2.morphologyEx(img_sevseg_red, cv2.MORPH_OPEN, kernel)
img_sevseg_red = cv2.bitwise_not(img_sevseg_red)

digits_rect = [(18, 18), (122, 18), (226, 18), (330, 18), (434, 18)]
#cv2.imshow("hhh", img_sevseg[18:18+124,226-6:226+92+6])
# recognize digits here

digit_imgs = []
for x,y in digits_rect:
	buf = img_sevseg_red[y:y+124,x-6:x+92+6]
	buf = cv2.resize(buf,(24,24))
	buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
	buf = buf.reshape([-1])
	buf = 255 - buf
	buf = np.float32(buf) / 255.
	digit_imgs.append(buf)

scr = conv.get_pred(digit_imgs)
print(scr)

#img_sevseg = cv2.inRange(img_sevseg, (0,0,200), (200,100,255))
#img_secvseg = cv2.inRange(img_sevseg, (0,0,200), (180,180,255))

#img_sevseg_hsv = cv2.inRange(img_sevseg_hsv, (0,200,200), (179,255,255))

#img_sevseg_red = cv2.inRange(img_sevseg_red, 200, 255)
#img_sevseg_red = cv2.adaptiveThreshold(img_sevseg_red,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#img_sevseg_red = img_sevseg[:,:]

#cv2.imshow('aaa',th_img)
cv2.imshow('bbb',cv2.resize(img,(800,600)))
#cv2.imshow('ddd', cv2.resize(dst,(800,600)))
cv2.imshow('eee', img_sevseg_red)
elapsed_time = time.time() - start_time
print(elapsed_time)

cv2.waitKey(0)
