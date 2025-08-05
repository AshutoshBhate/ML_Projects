import os
import numpy as np
import cv2
import pickle
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def train_model(data_dir="Photos_dl"):
    embedder = FaceNet()
    faces = []
    labels = []
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            embedding = embedder.embeddings(np.expand_dims(img, axis=0))[0]
            faces.append(embedding)
            labels.append(person_name)
    
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(faces, labels_encoded)
    
    with open("faces_dl_model.pkl", "wb") as f:
        pickle.dump({"encoder": encoder, "model": model}, f)
    
    print(f"Training complete. {len(faces)} samples processed.")

if __name__ == "__main__":
    train_model()