import data_retriver
import robot_prop 
import time
import util
import detection_mod
from camera_module import camera_thread
import armor_plate_mod
import rune_mod

data_reader = data_retriver.data_reader_thread()
data_reader.start()

camera_thread = camera_thread()
camera_thread.start()

while True:
	mode = robot_prop.mode
	if mode==1:
		armor_plate_mod.run(camera_thread)
	elif mode==0:
		time.sleep(0.1)
	else:
		error=rune_mod.run(camera_thread)
