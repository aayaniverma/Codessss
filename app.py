from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

vid = cv2.VideoCapture(0)

def generate_frames():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                print("No frame captured!")
                break
            
    
            print("Processing frame...")
            
            # Convert the image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = holistic.process(image_rgb)
            
            # Draw face landmarks
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # Draw right hand landmarks
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Draw left hand landmarks
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # Convert image back to BGR for displaying with OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
                        
# Route to render the web.html template
@app.route('/')
def web():
    return render_template('web.html')

# Route to render the product4.html template
@app.route('/product4')
def product4():
    return render_template('product4.html')

# Route to provide the video feed with generated frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)