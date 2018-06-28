import digit_detection
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
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
		coord, handwritten_num, Flaming_digit,handwritten_coords = digit_detection.get_digits2(image)
	except:
		time.sleep(0.1)
		continue
	print " im here"
	if coord == False:
		continue


	for coord in handwritten_coords:
		cv2.circle(image,coord,5,(0,255,0),-1)

	cv2.imshow('',image)
	cv2.waitKey(1)

	print  handwritten_num
