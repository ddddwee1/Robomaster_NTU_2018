def sec2hms(sec):
	hm = sec//60
	s = sec%60
	h = hm // 60
	m = hm % 60
	return h,m,s