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
counter_coord = 0
Target_lock=0

while True:

	mode = robot_prop.mode
	if mode==1:
		counter_coord, Target_lock=armor_plate_mod.run(camera_thread,counter_coord,Target_lock)
		print Target_lock
	elif mode==0:
		time.sleep(0.1)
	else:
		rune_mod.run(camera_thread)

