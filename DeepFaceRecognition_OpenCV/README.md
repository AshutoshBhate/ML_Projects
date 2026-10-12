# DeepFaceRecognition_OpenCV 

A **deep learning-based face recognition system** using OpenCV, FaceNet embeddings, and SVM classification. It recognizes faces in a live webcam feed, with optional GPU acceleration, and replaces the classic Haar cascade + LBPH approach with a deep learning pipeline.

## Key Features 
- **Deep Learning Face Detection**: Uses OpenCV's pretrained SSD (Single Shot MultiBox Detector) face detector, with a ResNet-10 backbone, through the DNN module.
- **FaceNet Embeddings**: Extracts 512-D face features using a pretrained FaceNet model (`keras-facenet`).
- **SVM Classifier**: A linear SVM with probability outputs, so every prediction comes with a confidence score.
- **Live Recognition**: Webcam feed with bounding boxes, names and confidence scores.
- **GPU Acceleration**: Runs face detection on CUDA when OpenCV is built with CUDA, and TensorFlow runs FaceNet on an NVIDIA GPU when one is available.

## Installation 

### Dependencies
- Python 3.8+
- OpenCV (`opencv-python`)
- Keras-FaceNet (`keras-facenet`)
- scikit-learn (`scikit-learn`)
- TensorFlow (`tensorflow`)

```bash
pip install opencv-python keras-facenet scikit-learn tensorflow
```

### Model Files
- The SSD face detector files, `deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel`, are included in this folder.
- `keras-facenet` downloads the FaceNet weights the first time it runs.
- The `opencv-python` package from pip is built without CUDA, so face detection runs on the CPU unless you use an OpenCV build with CUDA.

## Usage

### 1. Capture Faces
Run to collect face samples from the webcam (saves to `Photos_dl/[name]`):
```bash
python face_capture_dl.py
# Enter name when prompted (e.g., "Elon_Musk")
```
It saves 200 face crops per person. Press Q to stop early.

### 2. Train Model
Train the SVM classifier using FaceNet embeddings:
```bash
python face_train_dl.py
# Output: faces_dl_model.pkl (saved classifier)
```

### 3. Run Recognition
Start live recognition from the webcam. It uses CUDA for face detection when OpenCV has it and falls back to the CPU otherwise. Press Q to quit.
```bash
python face_recognition_dl_gpu.py
```

## Technical Overview 
1. **Face Detection**:  
   - SSD detector (300x300 input) with 90% confidence threshold.  
   - When capturing training faces, the bounding box is expanded by 20% (10% on each side) before cropping.

2. **Feature Extraction**:  
   - Each face is resized to 160x160, and FaceNet turns it into a 512-dimensional embedding.

3. **Classification**:  
   - A linear SVM predicts the identity, with a probability for each enrolled person.  
   - "Unknown" label if confidence < 95%.

## Customization 
- **Adjust Confidence Thresholds**: Modify `confidence >= 0.95` (recognition) in `face_recognition_dl_gpu.py`, or `confidence > 0.9` (face detection) in both scripts.
- **GPU/CPU Mode**: Toggle `use_gpu=True/False` in `FaceRecognizer()`.
- **Dataset Size**: Change `num_samples=200` in `face_capture_dl.py`.

## Why Deep Learning Over Haar/LBPH?
| Feature                | Haar cascade + LBPH                  | This project                                           |
|------------------------|--------------------------------------|--------------------------------------------------------|
| **Face Detection**     | Haar cascade (hand-crafted features) | SSD with a ResNet-10 backbone                          |
| **Face Features**      | LBPH texture histograms              | 512-D FaceNet embeddings learned from millions of faces |
| **Identification**     | Closest LBPH histogram               | Linear SVM with probabilities                          |
| **GPU Support**        | No                                   | CUDA for detection, TensorFlow GPU for FaceNet         |

Learned embeddings cope with changes in pose and lighting much better than LBPH histograms.
