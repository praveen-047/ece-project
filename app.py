import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow warnings
import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
from twilio.rest import Client

# Twilio setup
TWILIO_ACCOUNT_SID = ''  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = ''  # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = ''  # Replace with your Twilio phone number
TO_PHONE_NUMBER = ''  # Replace with recipient's phone number

# Function to send SMS alert via Twilio
def send_sms_alert():
    try:
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # Send SMS
        message = client.messages.create(
            body="The driver has been detected as drowsy multiple times. Please contact them to ensure their safety.",
            from_=TWILIO_PHONE_NUMBER,
            to=TO_PHONE_NUMBER
        )

        print("SMS sent successfully!")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(landmarks, eye_indices):
    points = np.array([landmarks[i] for i in eye_indices])
    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Indices for eyes (specific to Mediapipe face mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
EAR_THRESHOLD = 0.22  # Eye Aspect Ratio threshold for drowsiness
CONSEC_FRAMES = 20  # Number of consecutive frames for drowsiness detection

# Variables
frame_counter = 0
drowsy_count = 0  # Tracks the number of drowsy alerts

# Start video capture
cap = cv2.VideoCapture(1)

# Function to draw circle around the eyes
def draw_eye_circle(eye_indices, landmarks, frame, ear_color):
    # Get eye landmarks
    eye_points = np.array([landmarks[i] for i in eye_indices])

    # Calculate bounding box for the eye
    x_min = min(eye_points[:, 0])
    x_max = max(eye_points[:, 0])
    y_min = min(eye_points[:, 1])
    y_max = max(eye_points[:, 1])

    # Calculate the center and radius of the circle
    center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
    radius = int(max(x_max - x_min, y_max - y_min) // 2)

    # Draw the circle
    cv2.circle(frame, center, radius, ear_color, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark coordinates
            landmarks = []
            h, w, _ = frame.shape
            for lm in face_landmarks.landmark:
                landmarks.append((int(lm.x * w), int(lm.y * h)))

            # Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2

            # Determine the color based on EAR
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                ear_color = (0, 0, 255)  # Red for drowsiness
                if frame_counter >= CONSEC_FRAMES:
                    drowsy_count += 1
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    playsound(r"C:\Users\praveen.s\PycharmProjects\PythonProject2\mixkit-censorship-beep-long-1083.wav")
                    frame_counter = 0  # Reset counter after alert

                    # Send SMS if drowsy 3 times
                    if drowsy_count == 3:
                        send_sms_alert()
                        drowsy_count = 0  # Reset after sending SMS
            else:
                frame_counter = 0
                ear_color = (0, 255, 0)  # Green for active (no drowsiness)

            # Draw EAR on frame with dynamic color
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)

            # Draw circles around both eyes
            draw_eye_circle(LEFT_EYE, landmarks, frame, ear_color)
            draw_eye_circle(RIGHT_EYE, landmarks, frame, ear_color)

    # Display the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

