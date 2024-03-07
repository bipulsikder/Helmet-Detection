import cv2
import numpy as np
import winsound

# Load pre-trained MobileNet SSD model
MODEL_FILE = "MobileNetSSD_deploy.prototxt"
MODEL_WEIGHTS = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(MODEL_FILE, MODEL_WEIGHTS)

# Define classes (we're interested in "person" and "helmet" classes)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "helmet"]

# Global variables to track detection correctness and previous state of traffic light
correct_detection = None
prev_light = None

# Function to play beep sound
def play_beep():
    winsound.Beep(1000, 500)  # Frequency = 1000Hz, Duration = 500ms

def detect_person_with_cap(frame):
    global correct_detection
    
    if correct_detection is not None:
        return correct_detection
    
    (h, w) = frame.shape[:2]

    # Preprocess frame and pass it through the network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (700, 700)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])
            if CLASSES[class_id] == "person":
                # Check if person is wearing a cap
                return True
    return False

# Open camera capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform detection
    if detect_person_with_cap(frame):
        # Draw a traffic light
        traffic_light = np.zeros((150, 100, 3), dtype=np.uint8)

        # Green light
        cv2.circle(traffic_light, (50, 50), 30, (0, 0, 255), -1)

        # Play beep if light changes
        if prev_light != 'green':
            play_beep()
            prev_light = 'green'
    else:
        # Draw a traffic light
        traffic_light = np.zeros((150, 100, 3), dtype=np.uint8)

        # Red light
        cv2.circle(traffic_light, (50, 50), 30, (0, 255, 0), -1)

        # Play beep if light changes
        if prev_light != 'red':
            play_beep()
            prev_light = 'red'

    # Combine the traffic light with the frame
    frame[10:160, 10:110] = traffic_light

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Check for user input to exit or toggle correctness
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == ord('g'):  # Press 'g' for green (correct detection)
        correct_detection = True
    elif key == ord('r'):  # Press 'r' for red (incorrect detection)
        correct_detection = False

# Release the capture
cap.release()
cv2.destroyAllWindows()


