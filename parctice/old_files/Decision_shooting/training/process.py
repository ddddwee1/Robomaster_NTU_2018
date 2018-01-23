def getQ(l):
	Q = 0
	res = []
	for i in range(len(l)):
		current_item = l[-1-i].split(',')
		if i==0:
			Q = float(current_item[-1])
		else:
			Q = float(current_item[-1]) + 0.99*Q
		res.insert(0,l[-1-i]+','+str(Q))
	return res

def get_all(l):
	res = []
	for i in l:
		res.append(getQ(i))
	return res

def processFile(fileStr):
	st = fileStr.split('\n')
	st = st[:-1]
	res = []
	buffres = []
	for i in st:
		a = i.split(',')
		if a[-1]=='0.05':
			a[-1] = '0.01'
		if a[-1]=='-0.05':
			a[-1] = '-0.1'
		i = ','.join(a)
		if a[0]=='0' and a[1]=='0':
			buffres = []
			continue
		buffres.append(i)
		if a[-1]=='-1.0' or a[-1]=='1.05':
			if len(buffres)>10:
				res.append(buffres)
			buffres = []
	return res

filecnt = 0
f = open('list.txt')
for i in f:
	i = i.strip()
	f2 = open(i)
	fstr = f2.read()
	res = processFile(fstr)
	res = get_all(res)
	for j in res:
		fout = open('./training_piece/'+str(filecnt)+'.csv','w')
		fout.write('\n'.join(j)+'\n')
		fout.close()
		filecnt += 1