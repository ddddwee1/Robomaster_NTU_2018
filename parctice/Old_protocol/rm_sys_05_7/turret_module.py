import robot_prop
import threading
import time

class turret_thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def shoot(self):
		time.sleep(0.05)
	 	robot_prop.shoot = 1
		time.sleep(0.2)
		robot_prop.shoot = 0
		#time.sleep(0.001)
#	def shoot_armour(self):
#		robot.prop.shoot = 2
#	def stop_shooting(self):
#		robot.prop.shoot = 0
		

