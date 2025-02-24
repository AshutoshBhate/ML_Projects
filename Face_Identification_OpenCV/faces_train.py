import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Photos")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

# Walk through the "Photos" directory; each subfolder is for one person.
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            # Use the folder name as the label
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-")
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            
            # Open the image, convert to grayscale, and convert to a NumPy array
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# Save the label mappings for use during recognition
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# Train the recognizer on the extracted face regions and save the model
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

print("Training complete")
