import digit_detection2 #Take note this is not digit_detection
import cv2
import time
import numpy as np
from camera_module import camera_thread
from keyboard_module import keyboard_thread
import robot_prop


#Parameters
WHITE = (255,255,255)
BLACK = (0,0,0)
image_width = 1024
image_height = 768

#Initialize
camera = camera_thread() #30fps by default.
camera.start()
key_reader = keyboard_thread()
key_reader.start()
#TODO - may have to implement locks for multithreading
previous_key = None
pitch_angle = 0
yaw_angle = 0

while True:

	key = key_reader.read()

	#To hold the key value
	if key == '':
		if previous_key is not None:
			key = previous_key
		else:
			pass
	else:
		previous_key = key


	image = camera.read()
	if image is None:
		continue #no image
	else:
		cv2.imshow('',image)
		cv2.waitKey(1)
	try:
		coord, handwritten_num, Flaming_digit, handwritten_coords = digit_detection2.get_digits2(image)
	except:
		time.sleep(0.1)
		continue

	#Mark the handwritten digits in the original image. For verification purpose
	for coords in handwritten_coords:
		cv2.circle(image,coords,5,(0,255,0),-1)


	#Mark the center of the image with a blue dot
	cv2.circle(image,(image_width/2,image_height/2),5,(255,0,0),-1)
	cv2.imshow()
	cv2.waitKey(1)



	#find index of the desired digit in handwritten_num
	digit_index = None
	for i,digit in enumerate(handwritten_num):
		if digit == key:
			digit_index = i


	#TODO make the below code into a function to make the code more modular
	desired_coord = handwritten_coords[digit_index]
	current_pitch = robot_prop.t_pitch
	current_yaw = robot_prop.t_yaw
	x_delta = desired_coord[0] - image_width/2
	y_delta = desired_coord[1] - image_height/2
	if x_delta > 0:
		yaw_angle = int(float(current_yaw + 500)) #I am using 500 instead of 300 because the turret is not responsive to small increase in yaw angle when turning to the right
	elif x_delta <0:
		yaw_angle = int(float(current_yaw - 300))
	if y_delta > 0:
		pitch_angle = int(float(current_pitch + 400))
	elif y_delta <0:
		pitch_angle = int(float(current_pitch - 400))


	#Enforce angle limits
	if abs(yaw_angle) > 6000:
		yaw_angle = 6000
	if abs(pitch_angle) > 2000:
		pitch_angle = 2000


	#Set angles
	robot_prop.v2 = yaw_angle
	robot_prop.v1 = pitch_angle

	print  handwritten_num


