'''
Necessary installs

sudo apt-get update
sudo apt-get install python3-pip
pip3 install flask RPi.GPIO
pip3 install opencv-python opencv-contrib-python
pip3 install flask-cors

'''

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
'''
PI ONLY
import RPi.GPIO as GPIO
'''
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup GPIO for servo control
''' PI ONLY
SERVO_PIN = 17  # Adjust this to your GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz frequency
servo.start(0)  # Initialize servo
'''

# Load known face encodings and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

names = { 1: "Naod" }

# Function to move the servo to 180 degrees
'''PI ONLY
def turn_servo_180():
    servo.ChangeDutyCycle(12.5)  # Adjust for your servo
    time.sleep(1)
    servo.ChangeDutyCycle(0)    # Turn off signal
'''

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400

    file = request.files['image']

    # Convert the uploaded image to a NumPy array
    image = Image.open(file.stream)
    frame = np.array(image)

    # Convert the image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Your face detection code
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'status': 'error', 'message': 'No face detected'}), 400

    # Assume you're using LBPH face recognizer
    recognized = False  # Flag to check if the face was recognized
    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        id, confidence = recognizer.predict(face)  # Predict face ID and confidence

        # You can adjust this threshold based on your needs
        threshold = 100  # Lower values indicate a higher confidence
        if confidence < threshold:  # Recognized face (lower confidence = higher recognition)
            recognized = True
            # Return the recognized person's id and confidence
            '''
						turn_servo_180()
						'''
            return jsonify({'status': 'success', 'id': id, 'confidence': confidence}), 200
        else:
            # If the face is not recognized, handle it as unknown
            return jsonify({'status': 'error', 'message': 'Face not recognized. Confidence too low.'}), 400

    # If no faces are recognized, return an error
    if not recognized:
        return jsonify({'status': 'error', 'message': 'No recognized face found.'}), 400



# Clean up GPIO on exit
@app.teardown_appcontext
def cleanup(_):
    '''
    servo.stop()
    GPIO.cleanup()
    '''

# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
