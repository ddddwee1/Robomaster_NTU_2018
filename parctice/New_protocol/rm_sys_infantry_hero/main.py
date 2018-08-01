import data_retriver
import robot_prop 
import time
from camera_module import camera_thread
import armor_plate_mod
import rune_mod

data_reader = data_retriver.data_reader_thread()
data_reader.start()

camera_thread = camera_thread()
camera_thread.start()
Target_lock=0

while True:

	mode = robot_prop.mode
	if mode==1:
		Target_lock=armor_plate_mod.run(camera_thread,Target_lock)
	elif mode==0:
		time.sleep(0.1)
	else:
		rune_mod.run(camera_thread)

