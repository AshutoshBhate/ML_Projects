# CIFAR-10 Image Classification with CNN

This repository contains a Jupyter notebook that implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

##  Project Structure

- `Cifar_10.ipynb`: Main Jupyter notebook with all the code and explanations.
- No additional files or folders are required for this notebook to run.

##  Features

- Loads and preprocesses CIFAR-10 data using TensorFlow and Keras.
- Builds a custom CNN architecture.
- Trains the model and evaluates its performance on test data.
- Visualizes predictions and training history.
- Includes dropout and data normalization for improved accuracy.

##  Libraries Used

- TensorFlow
- Keras
- NumPy
- Matplotlib

Make sure to install these libraries using:

```bash
pip install tensorflow numpy matplotlib
```

## How to Run

### Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Launch Jupyter Notebook:

```bash
jupyter notebook Cifar_10.ipynb
```

### Run the notebook cells in sequence.

## Results

- Achieved over 71% accuracy on the test set. (Update with your final result!)
- Visualizations of predictions help evaluate model performance qualitatively.

## Future Improvements

- Implement data augmentation.
- Try different CNN architectures (e.g., ResNet, VGG).
- Use learning rate schedulers and early stopping.

