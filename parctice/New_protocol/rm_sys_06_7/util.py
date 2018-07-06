import numpy as np 
from math import sqrt
import cv2 

def get_delta(coord):
	if len(coord) == 0:
		return 0,0
	distance = [(x-320)+(y-240) for x,y,w,h in coord]
	min_index = np.argmax(-np.array(distance))
	target_coord = coord[min_index]
	x,y,w,h = target_coord
	
	# do process on x,y,w,h
	return int(float(y-240)*3000/240),int(float(x-320)*3000/320)

def get_delta_buf(x,y):

	# do process on x,y,w,h
	return int(float(y-240)*3000/240),int(float(x-320)*3000/320)


