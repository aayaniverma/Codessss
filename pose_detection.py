import mediapipe as mp
import cv2


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

vid = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while vid.isOpened():
        
        ret,frame = vid.read()
        img= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results= holistic.process(img)
        #print(results.face_landmarks)
        #face_landmarks,pose_landmarks
        #Recolour image for rendering 
        img= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
		#Drawing face landmarks
        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.POSE_CONNECTIONS)
        #Right hand 
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
		#POSE
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        



        cv2.imshow('Raw Webcam Feed',img)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()
