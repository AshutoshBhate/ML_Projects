import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
import tensorflow as tf

# GPU verification and configuration
print("OpenCV using CUDA:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
print("TensorFlow using GPU:", tf.test.is_gpu_available())
print("TensorFlow GPU Device:", tf.config.list_physical_devices('GPU'))

# Configure TensorFlow GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

class FaceRecognizer:
    def __init__(self, use_gpu=False):
        self.embedder = FaceNet()
        
        # Initialize face detector
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt.txt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        # GPU configuration
        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.gpu_enabled = True
            print("Using CUDA for face detection")
        else:
            self.gpu_enabled = False
            print("Using CPU for face detection")
            
        # Load classifier with error handling
        try:
            with open("faces_dl_model.pkl", "rb") as f:
                data = pickle.load(f)
                self.encoder = data["encoder"]
                self.model = data["model"]
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def recognize(self, frame):
        # Face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces_to_process = []
        face_boxes = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0]]*2)
                x1, y1, x2, y2 = box.astype(int)
                face = frame[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                    
                resized_face = cv2.resize(face, (160, 160))
                faces_to_process.append(resized_face)
                face_boxes.append((x1, y1, x2, y2))
        
        results = []
        if faces_to_process:
            embeddings = self.embedder.embeddings(np.array(faces_to_process))
            for i, embedding in enumerate(embeddings):
                probs = self.model.predict_proba([embedding])[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]
                
                # Confidence Check
                if confidence >= 0.95:
                    name = self.encoder.inverse_transform([best_idx])[0]
                else:
                    name = "Unknown"
                
                x1, y1, x2, y2 = face_boxes[i]
                results.append({
                    "box": (x1, y1, x2-x1, y2-y1),
                    "name": name,
                    "confidence": float(confidence)
                })
                    
        return results

if __name__ == "__main__":
    recognizer = FaceRecognizer(use_gpu=True)
    
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            faces = recognizer.recognize(frame)
            
            for face in faces:
                x, y, w, h = face["box"]
                label = f"{face['name']} {face['confidence']*100:.1f}%"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()