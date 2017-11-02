import numpy as np 

x1 = [2,3]
x2 = [0,0]
fout = open('data.csv','w')
for i in range(20):
	x = x1[0]+np.random.rand()
	y = x1[1]+np.random.rand()
	fout.write(str(x)+','+str(y)+',0\n')

for i in range(20):
	x = x2[0]+np.random.rand()
	y = x2[1]+np.random.rand()
	fout.write(str(x)+','+str(y)+',1\n')

fout.close()