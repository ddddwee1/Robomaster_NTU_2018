import numpy as np 

lr_bias = -1500.
fw_bias = 1500.

def get_order(ang,p0,p1):
	h0 = p0[0]
	w0 = p0[1]
	h1 = p1[0]
	w1 = p1[1]
	dh = h1-h0
	dw = w1-w0
	lst = [3,2,1,4,0,0,5,6,7]
	ind = (dh+1) * 3 + dw + 1
	d_ang = lst[ind]*5
	ang = (ang - d_ang) * 9
	lr = np.sin(ang*np.pi/180) * lr_bias
	fw = np.cos(ang*np.pi/180) * fw_bias
	lr = int(lr)
	fw = int(fw)
	if ind==4:
		return 0,0
	else:
		return lr,fw
