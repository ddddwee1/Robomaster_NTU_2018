import robot_prop
import threading
import time

class turret_thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def shoot(self):
		time.sleep(0.15)
	 	robot_prop.shoot = 1
		time.sleep(0.4)
		robot_prop.shoot = 0
		#time.sleep(0.001)

	def shoot_armour(self):

	 	robot_prop.shoot = 1
		time.sleep(0.1)
	 	#robot_prop.shoot = 0


