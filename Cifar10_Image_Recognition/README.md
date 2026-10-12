# CIFAR-10 Image Classification with CNN

This folder contains a Jupyter notebook that implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class (50,000 for training and 10,000 for testing).

##  Project Structure

- `Cifar_10.ipynb`: Main Jupyter notebook with all the code and explanations.
- No additional files or folders are required for this notebook to run. Keras downloads the dataset the first time it runs.

##  Features

- Loads CIFAR-10 using TensorFlow and Keras, and scales the pixel values to 0–1.
- Trains a fully connected network first, as a baseline.
- Builds a CNN: three convolution + max-pooling blocks (32, 64 and 128 filters), a 64-unit dense layer and a 10-class softmax output.
- Trains the CNN with Adam for 20 epochs and evaluates both models on the test data with a per-class classification report (precision, recall and F1).
- Shows sample images with their labels and checks a few predictions.

##  Libraries Used

- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

Make sure to install these libraries using:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## How to Run

### Clone the repository:

```bash
git clone https://github.com/AshutoshBhate/ML_Projects.git
cd ML_Projects/Cifar10_Image_Recognition
```

### Launch Jupyter Notebook:

```bash
jupyter notebook Cifar_10.ipynb
```

### Run the notebook cells in sequence.

## Results

| Model | Test accuracy |
| ----- | ------------- |
| Fully connected baseline (5 epochs) | 44% |
| CNN (20 epochs) | 70.9% |

- The CNN reaches 89.8% accuracy on the training set, so it overfits.
- Checking individual predictions against their images shows where the model gets classes right or wrong.

## Future Improvements

- Implement data augmentation and dropout to reduce overfitting.
- Try different CNN architectures (e.g., ResNet, VGG).
- Use learning rate schedulers and early stopping.
