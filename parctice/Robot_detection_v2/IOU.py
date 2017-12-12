def get_iou(inp1,inp2):	
	x1,y1,w1,h1 = inp1[0],inp1[1],inp1[2],inp1[3]
	x2,y2,w2,h2 = inp2[0],inp2[1],inp2[2],inp2[3]
	xo = min(abs(x1+w1/2-x2+w2/2), abs(x1-w1/2-x2-w2/2))
	yo = min(abs(y1+h1/2-y2+h2/2), abs(y1-h1/2-y2-h2/2))
	if abs(x1-x2) > (w1+w2)/2 or abs(y1-y2) > (h1+h2)/2:
		return 0
	elif abs(x1-x2) < abs(w1-w2):
		xo = min(w1, w2)
		if abs(y1-y2) < abs(h1-h2):
			yo = min(h1, h2)
	overlap = xo*yo
	total = w1*h1+w2*h2-overlap
	# print(overlap)
	# print(total)
	# print(iou)
	return overlap/total

get_iou([10,10,10,10],[10,10,1,1])
