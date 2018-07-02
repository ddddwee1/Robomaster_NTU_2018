import cv2
import numpy as np 

class video_saver():
    def __init__(self,name,size):
        self.name = name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.vidwriter = cv2.VideoWriter(name,fourcc,15.0,(size[1],size[0]))
    def write(self,img):
        self.vidwriter.write(img)
    def finish(self):
        self.vidwriter.release()

vid = video_saver('demo.mp4',(800,800))
for i in range(500):
	fname = './result/%d.png'%i
	img = cv2.imread(fname)
	vid.write(img)
vid.finish()