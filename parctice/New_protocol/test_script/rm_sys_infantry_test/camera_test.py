import camera_module2
from turret_module import turret_thread

turret_thread = turret_thread()
turret_thread.start()

camera_thread = camera_module2.camera_thread
camera_thread.start()

while True:
	turret_thread.shoot_armour()
