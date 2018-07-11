import digit_detection
import cv2
import time
import numpy as np

import time


def get_handwritten_coord(coord):
	x1 = (coord[2][0] + coord[2][2] + coord[3][0] +coord[3][2] - coord[0][0] - coord[1][0])//4 +  (coord[0][0] + coord[1][0]) //2
	y1 = (coord[1][1] + coord[1][3] + coord[3][1] +coord[3][3] - coord[0][1] - coord[2][1])//4 + (coord[0][1] + coord[2][1]) //2

	x0 = (x1 - coord[0][0])//2 + coord[0][0]
	y0 = (coord[0][1] +coord[0][3] + coord[2][1] + coord[0][3])//2

	x2 = (coord[2][0] + coord[2][2] - x1)//2 + x1
	y2 = (coord[1][1] + coord[3][1])//2

	handwritten_coord = [[x0,y0],[x1,y0],[x2,y0],[x0,y1],[x1,y1],[x2,y1],[x0,y2],[x1,y2],[x2,y2]]
	return handwritten_coord

def draw_handwritten_digits(image,handwritten_coord,radius):
	for x,y in handwritten_coord:
		cv2.circle(image,(x,y),radius, (0,0,255), 4)
	cv2.imshow('',image)
	cv2.waitKey(1)

def rune_shooting(image,saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted,bigbuff=False ):

	new_7seg = False

	try:
		coord,scr_7seg_raw, handwritten_num, Flaming_digit = digit_detection.get_digits(image,bigbuff)
		#print scr_7seg_raw, handwritten_num, Flaming_digit
	except:
		time.sleep(0.1)
		return [-1], saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted 

	if len(handwritten_num) == 1 and len(Flaming_digit) == 1 :
		cv2.imshow('',image)
		cv2.waitKey(1)
		print ("Wrong number")
		return [-1], saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted 

	if len(handwritten_num) != 1 and len(Flaming_digit) == 1 :
		num_9boxes = handwritten_num

	if len(handwritten_num) == 1 and len(Flaming_digit) != 1 :
		num_9boxes = Flaming_digit

	handwritten_coord = get_handwritten_coord(coord)

	#drawing for debugging
	radius = (coord[0][3]) //4
	draw_handwritten_digits(image, handwritten_coord, radius)

	# shooting logic
	for raw_7seg in scr_7seg_raw:
		if raw_7seg == 10:
			checked_7seg_raw = False
			break
		else: 
			checked_7seg_raw = True

	numbers_7seg = scr_7seg_raw[0]*10000 + scr_7seg_raw[1] * 1000+scr_7seg_raw[2] * 100 + scr_7seg_raw[3]*10+scr_7seg_raw[4]


	if checked_7seg_raw == False:
		return [-1], saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted 

	print saved_7seg_raw

	if saved_7seg_raw != numbers_7seg and checked_7seg_raw == True:
		saved_7seg_raw = numbers_7seg 
		scr_7seg_index = 0
		new_7seg = True
		whether_shooted = False

	if saved_7seg_raw == numbers_7seg and checked_7seg_raw == True:

		if scr_7seg_index == 0:
			num_7seg = scr_7seg_raw[scr_7seg_index]
			prev_num_7seg = scr_7seg_raw[scr_7seg_index]

		if scr_7seg_index == 5:
			num_7seg = scr_7seg_raw[scr_7seg_index-1]
			prev_num_7seg = scr_7seg_raw[scr_7seg_index-1]

		else:
			num_7seg = scr_7seg_raw[scr_7seg_index]
			prev_num_7seg = scr_7seg_raw[scr_7seg_index-1]

		saved_7seg_raw = numbers_7seg
		#print ('a',num_7seg)

	numbers_9boxes = num_9boxes[0]*10**8+num_9boxes[1]*10**7+num_9boxes[2]*10**6+num_9boxes[3]*10**5+num_9boxes[4]*10**4+num_9boxes[5]*10**3+num_9boxes[6]*10**2+num_9boxes[7]*10**1+num_9boxes[8]

	if saved_numbers_9boxes == -1 or saved_numbers_9boxes !=  numbers_9boxes or new_7seg == True:
		saved_numbers_9boxes = numbers_9boxes
		scr_7seg_index += 1
		whether_shooted = False
		for index in range(len(num_9boxes)):
			if num_7seg == num_9boxes[index]:
				num_9boxes_index = index
				print ('Next',prev_num_7seg,num_9boxes_index,scr_7seg_index)
				break

	else:
		#if num_9boxes_index != -1:
		print ('Before',prev_num_7seg,num_9boxes_index,scr_7seg_index)

	if scr_7seg_index >=6:
		scr_7seg_index = 0
		saved_numbers_9boxes = -1

	return handwritten_coord , saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted 
