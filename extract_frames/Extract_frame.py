import cv2

class data_reader():
	def __init__(self,fname):
		print('Reading data...')
		cap = cv2.VideoCapture(fname)
		cap.set(cv2.CAP_PROP_FPS,30)
		fps = cap.get(cv2.CAP_PROP_FPS)
		print 'fps',fps

		counter = 0
		counter_1 = 0

		while(cap.isOpened()):
			counter += 1
			ret, frame = cap.read()

			if counter <30:
				continue
			#if counter >800:
				#frame = cv2.resize(frame,(960,540))
			#cv2.imshow("cropped", img)
			#cv2.waitKey(0)
			if ret==True:
				if counter %2 == 0:
					cv2.imwrite('./frame/image'+str(counter_1)+'.png',frame)
					counter_1 += 1
			else:
				cap.release()


data_reader('red_training.webm')
