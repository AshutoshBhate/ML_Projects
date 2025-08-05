# DeepFaceRecognition_OpenCV 

A **deep learning-based face recognition system** using OpenCV, FaceNet embeddings, and SVM classification. Built for real-time recognition with optional GPU acceleration. Upgraded from traditional Haar/LBPH methods to a modern deep learning pipeline

## Key Features 
- **Deep Learning Face Detection**: Uses SSD (Single Shot MultiBox Detector) with OpenCV's DNN module.
- **FaceNet Embeddings**: Extracts 128-D face features using a pretrained FaceNet model.
- **SVM Classifier**: Robust classification with probabilistic confidence scores.
- **Real-Time Recognition**: Live webcam integration with bounding boxes and labels.
- **GPU Acceleration**: Supports CUDA for faster face detection and embeddings (NVIDIA GPUs).

## Installation 

### Dependencies
- Python 3.8+
- OpenCV (`opencv-python`)
- Keras-FaceNet (`keras-facenet`)
- scikit-learn (`scikit-learn`)
- TensorFlow (`tensorflow` or `tensorflow-gpu`)

```bash
pip install opencv-python keras-facenet scikit-learn tensorflow
```

### Additional Files
Download these pretrained models and place them in your project root:
1. **SSD Face Detector**:  
   - [deploy.prototxt.txt](https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detector/deploy.prototxt)  
   - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel)

## Usage

### 1. Capture Faces
Run to collect face samples (saves to `Photos_dl/[name]`):
```bash
python face_capture_dl.py
# Enter name when prompted (e.g., "Elon_Musk")
```

### 2. Train Model
Train the SVM classifier using FaceNet embeddings:
```bash
python face_train_dl.py
# Output: faces_dl_model.pkl (saved classifier)
```

### 3. Run Recognition
Start real-time recognition (use `--gpu` if available):
```bash
python face_recognition_dl_gpu.py
```

## Technical Overview 
1. **Face Detection**:  
   - SSD detector (300x300 input) with 90% confidence threshold.  
   - Bounding box expanded by 20% for better face cropping.

2. **Feature Extraction**:  
   - FaceNet model generates 128-dimensional embeddings from 160x160 face images.

3. **Classification**:  
   - SVM predicts identity with probability-based confidence.  
   - "Unknown" label if confidence < 95%.

## Customization 
- **Adjust Confidence Thresholds**: Modify `confidence >= 0.95` in `face_recognition_dl_gpu.py`.
- **GPU/CPU Mode**: Toggle `use_gpu=True/False` in `FaceRecognizer()`.
- **Dataset Size**: Change `num_samples=200` in `face_capture_dl.py`.

## Why Deep Learning Over Haar/LBPH?
| Feature               | Old (Haar/LBPH)       | New (Deep Learning)       |
|-----------------------|-----------------------|---------------------------|
| **Accuracy**          | Low on varied poses   | High (FaceNet embeddings) |
| **Lighting Robustness**| Poor                 | Excellent                 |
| **GPU Support**       | No                    | Yes                       |
| **Modern Practices**  | No                    | Industry-standard         |
