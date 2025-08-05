# MNIST Digit Recognition

This project is a simple machine learning model built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. It walks through loading the data, preprocessing it, building a neural network model, training it, and evaluating its performance.

##  Project Structure

- `MNIST_Digit_Recognition.ipynb`: Jupyter Notebook containing the complete workflow for digit recognition using the MNIST dataset.
- Uses TensorFlow and Keras for model building and training.

##  Features

- Loads and visualizes the MNIST dataset
- Preprocesses image data (normalization)
- Builds a fully connected neural network (Dense layers)
- Trains the model and evaluates its performance
- Visualizes predictions on test samples

##  Model Architecture

- Input Layer: Flatten (28x28 input pixels)
- Hidden Layers:
  - Dense layer with 128 neurons, ReLU activation
  - Dense layer with 64 neurons, ReLU activation
- Output Layer: Dense layer with 10 neurons (one per digit), Softmax activation

##  Accuracy

- The model achieves around **98%** accuracy on the training dataset and high accuracy on the test dataset as well.

##  Requirements

Make sure to install the following Python packages:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

### Clone the repository:

```
git clone https://github.com/your-username/MNIST-Digit-Recognition.git
cd MNIST-Digit-Recognition
```

### Open the notebook:

```
jupyter notebook MNIST_Digit_Recognition.ipynb
```

### Run the cells in sequence to train and test the model.

## Notes
- The model can be further improved using CNNs for better accuracy on image data.
- Currently uses a fully connected architecture suitable for beginners.
