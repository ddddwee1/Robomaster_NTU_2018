import data_retriver
import robot_prop 
import time
import util
from camera_module import camera_thread
#import armor_plate_mod
import rune_mod

data_reader = data_retriver.data_reader_thread()
data_reader.start()

camera_thread = camera_thread()
camera_thread.start()

while True:

	rune_mod.run(camera_thread)
