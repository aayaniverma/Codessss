from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load the t-shirt image
t_shirt_img = cv2.imread('image.jpeg', cv2.IMREAD_UNCHANGED)
print(t_shirt_img.shape)
if t_shirt_img is None:
    print(f"Error loading t-shirt image from 'C:\\Users\\Dell\\OneDrive\\Desktop\\Myntra-HackerRamp\\shirt-img.jpg'")
else:
    shirt_height, shirt_width, channels = t_shirt_img.shape
    t_shirt_center_x = shirt_width / 2
    t_shirt_center_y = shirt_height / 2

    # Remove the background from the t-shirt image
    def remove_background(image):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define range of background color in HSV
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 20, 255])
        # Threshold the HSV image to get only the background colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Invert the mask to get the foreground
        mask_inv = cv2.bitwise_not(mask)
        # Convert the mask to 3 channels
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        # Apply the mask to the image
        result = cv2.bitwise_and(image, mask_inv)
        # Add alpha channel to the result
        b, g, r = cv2.split(result)
        a = mask_inv[:, :, 0]
        result = cv2.merge([b, g, r, a])
        return result

    t_shirt_img = remove_background(t_shirt_img)

vid = cv2.VideoCapture(0)

def generate_frames():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                print("No frame captured!")
                break
            
            print("Processing frame...")
            
            # Convert the image from BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = holistic.process(image_rgb)
            
            # Draw face landmarks
            mp_drawing.draw_landmarks(image_rgb, results.face_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # Draw right hand landmarks
            mp_drawing.draw_landmarks(image_rgb, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Draw left hand landmarks
            mp_drawing.draw_landmarks(image_rgb, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Draw pose landmarks and overlay t-shirt image
            if results.pose_landmarks and t_shirt_img is not None:
                landmarks = results.pose_landmarks.landmark
                shoulder_left = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
                shoulder_right = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
                hip_left = landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value]
                hip_right = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value]

                x_coords = [shoulder_left.x, shoulder_right.x,hip_left.x, hip_right.x]
                y_coords = [shoulder_left.y, shoulder_right.y,hip_left.y, hip_right.y]

                x_avg = sum(x_coords) / len(x_coords)
                y_avg = sum(y_coords) / len(y_coords)

                # Calculate the offset between the center of the body and the center of the t-shirt image
                offset_x = int(max(0, min(x_avg * frame.shape[1] - t_shirt_center_x, frame.shape[1] - shirt_width)))
                offset_y = int(max(0, min(y_avg * frame.shape[0] - t_shirt_center_y, frame.shape[0] - shirt_height)))

                # Overlay the t-shirt image on the webcam feed
                roi = image_rgb[offset_y:offset_y+shirt_height, offset_x:offset_x+shirt_width]
                alpha_t_shirt = t_shirt_img[:, :, 3] / 255.0
                alpha_frame = 1.0 - alpha_t_shirt

                for c in range(0, 3):
                    roi[:, :, c] = (alpha_t_shirt * t_shirt_img[:, :, c] + alpha_frame * roi[:, :, c])

            # Convert the image back to BGR for displaying with OpenCV
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def web():
    return render_template('web.html')

@app.route('/product4')
def product4():
    return render_template('product4.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
