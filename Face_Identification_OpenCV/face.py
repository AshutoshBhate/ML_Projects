import numpy as np
import cv2
import pickle

# Load cascades
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

# Load recognizer and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
with open('labels.pickle', 'rb') as f:
    label_ids = pickle.load(f)
    labels = {v: k for k, v in label_ids.items()}

cap = cv2.VideoCapture(0)

# Define our mapping parameters
min_conf = 80  # best case scenario
max_conf = 110  # worst acceptable match

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        # Determine if recognized based on a threshold:
        if conf < 115:
            name = labels.get(id_, "unknown")
        else:
            name = "unknown"

        # Map the LBPH distance (conf) to a pseudo-percentage
        if conf < min_conf:
            conf = min_conf
        elif conf > max_conf:
            conf = max_conf

        confidence_percentage = int(round((max_conf - conf) / (max_conf - min_conf) * 100))
        
        # Debug output:
        print("ID:", id_, "Name:", name, "Distance:", conf, "Confidence:", confidence_percentage, "%")
        
        # Display the label and confidence percentage
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{name} {confidence_percentage}%", (x, y - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Optionally draw eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
