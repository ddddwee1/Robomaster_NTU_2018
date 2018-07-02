import numpy as np 
import time 

data = open('data.npy','rb')
data = np.load(data)
data = data.transpose()

group_length = data.shape[1]
coord = open('coord.npy','rb')
coord = np.load(coord)
print(group_length)

bufdata = [data]
for i in range(1,40):
	bufdata.append(np.roll(data,i*3,axis=0))

data = np.concatenate(bufdata,axis=1)

def get_coord2(a):
	t1 = time.time()

	dt = data.copy()
	a = np.float32(a).reshape([-1,1])
	non0_a = a!=0
	non0_dt = dt!=0

	t2 = time.time()

	non0_all = non0_a * non0_dt

	t3 = time.time()

	b = (a - dt)*non0_all
	b = np.abs(b)

	t4 = time.time()

	b = np.sum(b,axis=0)/(non0_all).sum(axis=0)	

	ind = np.argmax(-b)

	t5 = time.time()

	ang = ind//group_length
	ind = ind%group_length
	hw = coord[ind]
	h = hw[0]
	w = hw[1]
	return ang,h,w
