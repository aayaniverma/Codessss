import cv2 
import numpy as np

body_detector = cv2.CascadeClassifier("haarcascade_upperbody.xml")
vid = cv2.VideoCapture(0) 
while(True):  
	ret, frame = vid.read() 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	results = body_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	for (x, y, w, h) in results:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		 
	cv2.imshow('frame', frame) 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

vid.release() 
cv2.destroyAllWindows()