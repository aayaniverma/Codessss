import cv2 
import numpy as np
# from matplotlib import pyplot as plt
body_detector = cv2.CascadeClassifier('../Downloads/haarcascade_upperbody.xml')
vid = cv2.VideoCapture(0) 
img = cv2.imread("../Downloads/pngegg.png")
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray, 1,255,cv2.THRESH_BINARY)
orb = cv2.ORB()
alpha = 0
while(True):  
	ret, frame = vid.read() 
	# cv2.rectangle(frame, (550, 260), (700, 500), (0,255,0), 2)
	frame = cv2.flip(frame, 1)
	# roi = frame[0:500, 0:500]
	# roi[np.where(mask)]= 0
	# roi += img
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	results = body_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	for (x, y, w, h) in results:
		roi = frame[x:y, x+w: y+h]
		print(np.where(mask))
		roi[np.where(mask)] = 0
		roi += img
	# 	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)	 
	cv2.imshow('frame', frame)
	# plt.imshow(frame),plt.show()
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
	if cv2.waitKey(1) == ord('a'):
		alpha +=0.1
		if alpha >=1.0:
			alpha = 1.0

vid.release() 
cv2.destroyAllWindows() 
