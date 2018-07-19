from socket import *
import threading 
import numpy as np 
import robot_prop
import util

class network_thread(threading.Thread):
	def __init__(self):
		self.socket = socket(AF_INET, SOCK_STREAM)
		self.socket.bind(('',8080))
		self.socket.listen(1)
		threading.Thread.__init__(self)

	def run(self):
		conn, addr = self.socket.accept()
		while True:
			data = conn.recv(1)
			if data==b'\xaa':
				data = conn.recv(1)
				if data == b'\xcc':
					data = conn.recv(6)
					w,h,ang = util.decode_whang(data)
					robot_prop.r2h = h
					robot_prop.r2w = w 
					robot_prop.r2rot = ang 
					datapack = b'\xaa\xcc'
					h,w,ang = robot_prop.get_current_pos()
					datapack += util.encode_whang(w,h,ang)
					conn.send(datapack)