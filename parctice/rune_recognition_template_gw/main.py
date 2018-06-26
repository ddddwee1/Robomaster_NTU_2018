import digit_detection
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

while True:

	_,image = cap.read()
	if image is None:
		break
	try:
		coord,scr_7seg_raw, handwritten_num, Flaming_digit = digit_detection.get_digits(image)
	except:
		time.sleep(0.1)
		continue

	if coord == False:
		continue


	#x0 = 
	x1 = (coord[2][0] + coord[2][2] + coord[3][0] +coord[3][2] - coord[0][0] - coord[1][0])//4 +  (coord[0][0] + coord[1][0]) //2
	y1 = (coord[1][1] + coord[1][3] + coord[3][1] +coord[3][3] - coord[0][1] - coord[2][1])//4 + (coord[0][1] + coord[2][1]) //2

	x0 = (x1 - coord[0][0])//2 + coord[0][0]
	y0 = (coord[0][1] +coord[0][3] + coord[2][1] + coord[0][3])//2

	x2 = (coord[2][0] + coord[2][2] - x1)//2 + x1
	y2 = (coord[1][1] + coord[3][1])//2

	radius = (coord[0][3]) //4

	handwritten_coord = [[x0,y0],[x1,y0],[x2,y0],[x0,y1],[x1,y1],[x2,y1],[x0,y2],[x1,y2],[x2,y2]]
	for x,y in handwritten_coord:
		cv2.circle(image,(x,y),radius, (0,0,255), 4)

	cv2.imshow('',image)
	cv2.waitKey(1)

	print scr_7seg_raw, handwritten_num
