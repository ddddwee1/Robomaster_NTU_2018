import numpy as np
import cv2
import conv
import math
import time
import random
import sys
import matplotlib.pyplot as plt

start_time = time.time()
# TODO: Adjustable Parameters
TEST_FILE_NAME = 'C.jpg'
# Resizing the original image or not before processed it
RESIZING_PROCESSED_IMAGE = True
# In cnt_hier_classifier, to detemine the minimum threshold for the amount of
# contour in a level
CONTOUR_LEVEL_MINIMUM_CONTOUR_AMOUNT = 10
# In centroid_linearity_test, this is the parameters for hough line transform
# TODO: Please adjust this parameters depending on the situation on site
HOUGH_LINE_TRANSFORM_THRESHOLD = 4
HOUGH_LINE_RHO_RESOLUTION = 4
HOUGH_LINE_THETA_RESOLUTION = 0.5*np.pi/180
HOUGH_LINE_MIN_THETA = -20*np.pi/180
HOUGH_LINE_MAX_THETA = 20*np.pi/180
# CONSTANT
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255,255,255)
EPSILON = 0.00001

def cnt_hier_classifier(current_index, min_cnt_amt, all_hiers, all_cnts):
	"""
	This is the function that seperates the contours based on their level in hierarchy
	"""
	cnt = all_cnts[current_index]
	hier = all_hiers[current_index]
	current_cnt_levels = []
	if hier[0] != -1:
		current_cnt_levels = cnt_hier_classifier(hier[0], min_cnt_amt, all_hiers, all_cnts)
	else:
		if hier[2] == -1:
			current_cnt_levels = [[current_index]]
		else:
			current_cnt_levels = [[]]
	if hier[2] != -1:
		current_cnt_levels.extend(cnt_hier_classifier(hier[2], min_cnt_amt, all_hiers, all_cnts))
	else:
		current_cnt_levels[0].append(current_index)

	# Remove any contour levels that has less than min_cnt_amt
	if hier[1] == -1:
		if len(current_cnt_levels[0]) < min_cnt_amt:
			current_cnt_levels.pop(0)
	return current_cnt_levels

def points_linearity_test(points, img_shape, threshold, rho_res, theta_res, min_theta, max_theta):
	"""
	This is the function that checks if some of the points in points form a line.
	This is based on the Hough Line Transform.
	"""
	max_length = math.sqrt(img_shape[0]**2 + img_shape[1]**2)
	row = int(max_length / rho_res) + 1
	colomn = int((max_theta - min_theta) / theta_res) + 1
	polling_matrix = np.zeros((row,colomn), dtype=int)
	for point in points:
		for i in range(colomn):
			theta = i*theta_res + min_theta
			rho = point[0]*math.cos(theta) + point[1]*math.sin(theta)
			polling_matrix[int(rho/rho_res)][i] += 1
	return np.argwhere(polling_matrix > threshold)

def get_detection_rune(img_ori):
	"""
	Main Process
	"""
	# img_ori = cv2.imread(TEST_FILE_NAME)
	# Enlarge the image or not
	if RESIZING_PROCESSED_IMAGE == True:
		img = cv2.resize(img_ori, (800,800), interpolation = cv2.INTER_LINEAR)
	else:
		img = img_ori.copy()
	# Convert to grayscale
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Blur the image to remove noises
	img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
	# Using otsu's binarization to threshold the image and find the scoreboard rectangle
	ret, th_img = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imshow('', th_img)
	# Find the contours from the thresholded image
	_, cnts, hier = cv2.findContours(th_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Sort the contours based on their level in the hierarchy
	contour_levels = cnt_hier_classifier(0, CONTOUR_LEVEL_MINIMUM_CONTOUR_AMOUNT, hier[0], cnts)

	# Do linearity test based on the centroid and bounding rect top left of each contours
	side_rects_centroid = []
	side_rects_bnd_rect = []
	img_shape = img.shape
	for level in contour_levels:
		contours = []
		centroid = []
		top_left = []
		bnd_rect = []
		for cnt_index in level:
			c = cnts[cnt_index]
			M = cv2.moments(c)
			if M["m00"] < 0.001:
				continue
			# Find the centroid of the contours
			contours.append(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			centroid.append([cX,cY])
			# Find the top left of the bounding rect
			x,y,w,h = cv2.boundingRect(c)
			top_left.append([x,y])
			bnd_rect.append([x,y,w,h])

		centroid_lines = points_linearity_test(centroid, img_shape\
												, HOUGH_LINE_TRANSFORM_THRESHOLD\
												, HOUGH_LINE_RHO_RESOLUTION\
												, HOUGH_LINE_THETA_RESOLUTION\
												, HOUGH_LINE_MIN_THETA\
												, HOUGH_LINE_MAX_THETA)
		top_left_lines = points_linearity_test(top_left, img_shape\
												, HOUGH_LINE_TRANSFORM_THRESHOLD\
												, HOUGH_LINE_RHO_RESOLUTION\
												, HOUGH_LINE_THETA_RESOLUTION\
												, HOUGH_LINE_MIN_THETA\
												, HOUGH_LINE_MAX_THETA)
		if len(centroid_lines) == 0:
			continue
		if len(top_left_lines) == 0:
			continue
		centroid_line_points = [[] for i in range(len(centroid_lines))]
		top_left_line_points = [[] for i in range(len(top_left_lines))]

		for i in range(len(contours)):
			cnt = contours[i]
			ctr = centroid[i]
			tpl = top_left[i]
			for j in range(len(centroid_lines)):
				theta = centroid_lines[j][1]
				theta = (theta*HOUGH_LINE_THETA_RESOLUTION) + HOUGH_LINE_MIN_THETA
				rho = ctr[0]*math.cos(theta) + ctr[1]*math.sin(theta)
				rho = int(rho / HOUGH_LINE_RHO_RESOLUTION)
				if rho == centroid_lines[j][0]:
					centroid_line_points[j].append(i)

			for j in range(len(top_left_lines)):
				theta = top_left_lines[j][1]
				theta = (theta*HOUGH_LINE_THETA_RESOLUTION) + HOUGH_LINE_MIN_THETA
				rho = tpl[0]*math.cos(theta) + tpl[1]*math.sin(theta)
				rho = int(rho / HOUGH_LINE_RHO_RESOLUTION)
				if rho == top_left_lines[j][0]:
					top_left_line_points[j].append(i)

		#print(top_left_line_points)
		#print(centroid_line_points)
		for points in top_left_line_points:
			if points in centroid_line_points:
				for idx in points:
					side_rects_centroid.append(centroid[idx])
					side_rects_bnd_rect.append(bnd_rect[idx])

	center = np.array(side_rects_centroid)
	center = center[:,0]
	center = np.mean(center)
	sr_tl = [0,10000]
	sr_bl = [0,0]
	sr_tr = [0,10000]
	sr_br = [0,0]
	for centroid in side_rects_centroid:
	    if centroid[0] < center:
	        if centroid[1] < sr_tl[1]:
	            sr_tl = centroid

	        if centroid[1] > sr_bl[1]:
	            sr_bl = centroid

	    elif centroid[0] > center:
	        if centroid[1] < sr_tr[1]:
	            sr_tr = centroid

	        if centroid[1] > sr_br[1]:
	            sr_br = centroid

	pts1 = np.float32([sr_tl,sr_tr,sr_bl,sr_br])
	pts2 = np.float32([[50, 280],[1390, 280],[50, 800],[1390, 800]])
	pts2 = pts2 + np.float32([70, 35])
	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(img,M,(1580,1050))

	for rect in side_rects_bnd_rect:
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
		buf = dst[y+10:y+150,x+15:x+265]
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
	    cv2.rectangle(dst,(x,y),(x+280,y+160),(0,0,255),3)
	    cv2.putText(dst,str(s),(x,y+50), FONT, 2,(0,0,255),2,cv2.LINE_AA)

	img_sevseg = dst[55:217,518:1062]

	img_sevseg_red = img_sevseg[:,:,2].copy()
	img_sevseg_float = img_sevseg.copy()
	img_sevseg_float = img_sevseg_float.astype(float)
	img_sevseg_float = img_sevseg_float / 256
	img_sevseg_avg = np.average(img_sevseg, axis = 2, weights=[1./2, 1./4, 1./4])
	img_sevseg_avg = img_sevseg_avg.astype(int)
	img_sevseg_var = np.var(img_sevseg_float, axis = 2)

	img_sevseg_red = cv2.inRange(img_sevseg_red, 210, 255)
	kernel = np.ones((7,7),np.uint8)

	img_sevseg_red = cv2.morphologyEx(img_sevseg_red, cv2.MORPH_OPEN, kernel)
	img_sevseg_red = cv2.bitwise_not(img_sevseg_red)

	digits_rect = [(18, 18), (122, 18), (226, 18), (330, 18), (434, 18)]

	# recognize digits here

	digit_imgs = []
	for x,y in digits_rect:
		buf = img_sevseg_red[y:y+124,x-12:x+92+12]
		buf = cv2.resize(buf,(24,24),interpolation=cv2.INTER_AREA)
		buf = cv2.copyMakeBorder(buf, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
		# zeros = np.ones([28,28],dtype=np.uint8) * 255
		# zeros[3:3+24,3:3+24] = buf
		# buf = zeros
		cv2.imshow('adsaf',buf)
		cv2.waitKey(0)
		buf = buf.reshape([-1])
		buf = 255 - buf
		buf = np.float32(buf) / 255.
		digit_imgs.append(buf)

	scr_7seg = conv.get_pred(digit_imgs)
	print(scr)

	cv2.rectangle(dst,(517,55),(1062,218),(0,0,255),3)
	cv2.putText(dst,str(scr_7seg),(518,55+50), FONT, 2,(0,0,255),2,cv2.LINE_AA)

	#cv2.imshow('aaa',th_img)
	cv2.imshow('bbb',cv2.resize(img,(800,600)))
	cv2.imshow('ddd', cv2.resize(dst,(800,600)))
	cv2.imshow('eee', img_sevseg_red)
	elapsed_time = time.time() - start_time
	print(elapsed_time)

	cv2.waitKey(0)

if __name__=='__main__':
	im = cv2.imread('c.jpg')
	get_detection_rune(im)