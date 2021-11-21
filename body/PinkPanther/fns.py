def left_armpit(pos, convType, location):
	if convType == "sim2real":
		if location == "front":
			return int((((pos+0.3)/(0.6))*(-166))+583)
		elif location == "back":
			return int((((pos+0.3)/(0.6))*(166))+417)
	elif convType == "real2sim":
		if location == "front":
			return ((((pos-583)/(-166))*(0.6))-0.3)
		elif location == "back":
			return ((((pos-417)/(166))*(0.6))-0.3)

def right_armpit(pos, convType):
	if convType == "sim2real":
		return int((((pos+0.3)/(0.6))*(-166))+583)
	elif convType == "real2sim":
		return ((((pos-583)/(-166))*(0.6))-0.3)

def left_shoulder(pos, convType):
	if convType == "sim2real":
		return int((((pos+0.3)/(1.5))*(-398))+688)
	elif convType == "real2sim":
		return ((((pos-688)/(-398))*(1.5))-0.3)

def right_shoulder(pos, convType):
	if convType == "sim2real":
		return int((((pos+0.3)/(1.5))*(387))+313)
	elif convType == "real2sim":
		return ((((pos-313)/(387))*(1.5))-0.3)

def left_elbow(pos, convType):
	if convType == "sim2real":
		return int((((pos+0.9)/(2.1))*(500))+500)
	elif convType == "real2sim":
		return ((((pos-500)/(500))*(2.1))-0.9)

def right_elbow(pos, convType):
	if convType == "sim2real":
		return int((((pos+0.9)/(2.1))*(-500))+500)
	elif convType == "real2sim":
		return ((((pos-500)/(-500))*(2.1))-0.9)

