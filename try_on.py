import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#Load the t-shirt image with a transparent background
t_shirt_img = cv2.imread('../Desktop/shirt_img.png', cv2.IMREAD_UNCHANGED)
shirt_height, shirt_width, _ = t_shirt_img.shape
t_shirt_center_x = shirt_width / 2 
t_shirt_center_y = shirt_height / 2 

vid = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while vid.isOpened():
        
        ret, frame = vid.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder_left = landmarks[11]
            shoulder_right = landmarks[12]
            elbow_left = landmarks[13]
            elbow_right = landmarks[14]
            waist_left = landmarks[23]
            waist_right = landmarks[24]

            x_coords = [shoulder_left.x, shoulder_right.x, elbow_left.x, elbow_right.x, waist_left.x, waist_right.x]
            y_coords = [shoulder_left.y, shoulder_right.y, elbow_left.y, elbow_right.y, waist_left.y, waist_right.y]

            x_avg = sum(x_coords) / len(x_coords)
            y_avg = sum(y_coords) / len(y_coords)

            # Calculate the offset between the center of the body and the center of the 2D image
            offset_x = int(max(0, min(x_avg * frame.shape[1] - t_shirt_center_x, frame.shape[1] - shirt_width)))
            offset_y = int(max(0, min(y_avg * frame.shape[0] - t_shirt_center_y, frame.shape[0] - shirt_height)))

            # Overlay the t-shirt image on the webcam feed
            roi = frame[offset_y:offset_y+shirt_height, offset_x:offset_x+shirt_width]
            for c in range(0, 3):
                roi[:, :, c] = np.where(t_shirt_img[:, :, 3] == 0, roi[:, :, c], t_shirt_img[:, :, c])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('Raw Webcam Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
                break
vid.release()
cv2.destroyAllWindows()
