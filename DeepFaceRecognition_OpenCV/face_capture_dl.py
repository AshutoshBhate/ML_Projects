import cv2
import os
import numpy as np

def capture_faces(name, output_dir="Photos_dl", num_samples=200):
    save_path = os.path.join(output_dir, name)
    os.makedirs(save_path, exist_ok=True)
    
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt.txt", 
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, 
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        net.setInput(blob)
        detections = net.forward()
        
        display_frame = frame.copy()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                expand_ratio = 0.2
                width = x2 - x
                height = y2 - y
                pad_x = int(width * expand_ratio / 2)
                pad_y = int(height * expand_ratio / 2)
                
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                face = frame[y:y2, x:x2]
                
                if face.size == 0:
                    continue
                
                # Resize to FaceNet's expected input (160x160)
                face = cv2.resize(face, (160, 160))
                cv2.imwrite(os.path.join(save_path, f"{name}_{count}.jpg"), face)
                count += 1
                print(f"Captured sample {count}/{num_samples}")
                
                # Draw expanded bounding box
                cv2.rectangle(display_frame, (x, y), (x2, y2), (255, 0, 0), 2)
        
        cv2.putText(display_frame, f"Captured: {count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press Q to quit", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Face Capture - Press Q to Quit", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter person's name: ").strip()
    capture_faces(name)