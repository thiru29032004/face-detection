import cv2
import argparse
import winsound  # Only works on Windows for sound, can be replaced for other platforms

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Face Detection using OpenCV with Video/Audio")
parser.add_argument('--video', type=str, help='Path to video file (leave empty for webcam)')
parser.add_argument('--image', type=str, help='Path to image file')
args = parser.parse_args()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# If video path is provided, use it, else use webcam
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)  # 0 for webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Play sound when face is detected (Windows only)
        winsound.Beep(1000, 300)  # Frequency 1000Hz, duration 300ms
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
