import digit_detection
import rune_shooting_logic
import cv2
import time
import numpy as np

import time

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS,10)
fps = cap.get(cv2.CAP_PROP_FPS)
#print 'fps',fps

cap.set(14, 0.0)  #exposure
cap.set(10, 0.02) #brightness

image_width = 640
image_height = 480

cap.set(3, image_width);
cap.set(4, image_height);

scr_7seg_index = 0
saved_7seg_raw = -1
saved_numbers_9boxes = -1
num_9boxes_index = -1

whether_shooted = False

while True:
	_,image = cap.read()

	if image is None:
		break

	handwritten_coord, saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted = rune_shooting_logic.rune_shooting(image,saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted,bigbuff=True)

	#print 'abc', handwritten_coord , saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted

	if len(handwritten_coord)==1:
		continue

	print handwritten_coord,num_9boxes_index
