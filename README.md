
# Handwritten Digit Recognition using MATLAB

# Overview

This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset in MATLAB. It utilizes convolutional, pooling, ReLU, and Softmax layers to process input images and predict handwritten digits with high accuracy. The model is trained on the MNIST dataset and classifies new digit images using learned features. Below is an overview of the analysis, along with sample outputs and results. This project was done in May' 2023.


## Block Diagram

- The below block diagram gives an overview of the overall funtionality of the implemented project
<p align="center">
  <img src="https://i.postimg.cc/65WDVk02/Picture2.jpg" alt="App Screenshot" width="600">
</p>

- The below block diagram describes the relation between all the functions
<p align="center">
  <img src="https://i.postimg.cc/ZRm3gtVp/Picture3.jpg" alt="App Screenshot" width="600">
</p>


## Features

- **Preprocessing and Training with MNIST Dataset**: The system loads and preprocesses the MNIST dataset, which contains 60,000 training and 10,000 test images of handwritten digits. Data is normalized and batch-trained using backpropagation with gradient descent to enhance learning efficiency.

- **ADMM Integration**: This function implements Alternating Direction Method of Multipliers (ADMM) for training a Convolutional Neural Network (CNN) on the MNIST dataset. It updates convolutional, fully connected, and output layer weights using backpropagation, momentum-based optimization, and ADMM constraints. The shrinkage function applies soft-thresholding for weight regularization.


- **Handwritten Digit Classification**: After training, the model classifies handwritten digits with high accuracy. Input images are fed into the trained CNN, which predicts the digit using fully connected layers and the Softmax function. The system generalizes well across diverse handwriting styles.
Input image (Digit 2 - on Paint):
<p align="center">
  <img src="https://i.postimg.cc/W15SZ760/x1.png" alt="App Screenshot" width="300">
</p>

  a) Funtions perfomed on the input image:
  <p align="center">
    <img src="https://i.postimg.cc/K87VKLR2/x2.png" alt="App Screenshot" width="300">
    <img src="https://i.postimg.cc/jdfN44HG/x3.png" alt="App Screenshot" width="300">
  
  </p>
  
  <p align="center">
    <img src="https://i.postimg.cc/j5XnLR7c/x4.png" alt="App Screenshot" width="300">
    <img src="https://i.postimg.cc/dVNhMWKH/x5.png" alt="App Screenshot" width="300">
  
  </p>

## Functions and Scripts

### 1. `TestMnistConv()`

- This script serves as the entry point for the CNN implementation.
- Loads data and creates labels using `loadData()` and `loadLabels()` functions.
- Invokes `MnistConv()` to perform CNN for the MNIST dataset.
- Saves the resulting data in the 'MnistConv.mat' file for comparison.

### 2. `MnistConv()`

- Defines hyperparameters such as learning rate (`alpha`) and gradient descent with momentum (`beta`).
- Executes four epoch loops, running functions like `Conv()`, `ReLU()`, `Pool()`, and `Softmax()` for each iteration.
- Utilizes backpropagation to compute weight gradients.
- Computes layers for the input layer, hidden layer, and convolution layer.

### 3. `showMnist()`

- Displays a sample of the MNIST dataset, showcasing the first 25 samples.

### 4. `PlotFeatures()`

- Initiates the image comparison process, relying on the 'MnistConv.mat' file.
- Calls functions like `Conv()`, `ReLU()`, `Pool()`, and `Softmax()` for the comparison.
- Produces matrices for visualization.

### 5. `Conv()`

- Performs the convolution operation on input tensors and filter sets.
- Handles dimension adjustments and rotation.
- Considers only samples without zero-padding.

### 6. `ReLU()`

- Implements Rectified Linear Unit (ReLU) to introduce non-linearity into the neural network.
- Applied separately to the real and imaginary parts of the input.

### 7. `Pool()`

- Executes Maxpooling, multiplying a matrix with a smaller identity matrix.
- Filters and extracts maximum values into a new matrix.

### 8. `Softmax()`

- Converts input vectors into exponential values.
- Normalizes the resulting values.

### 9. `display_network()`

- Visualizes each filter type on the input.
- Resizes samples into smaller blocks and arranges them in a matrix for display.



- **Testing Results**: The results are displayed via 2 different funtions: 

  a) TestMnistConv: Displays the number of epochs and the accuracy
  <p align="center">
  <img src="https://i.postimg.cc/W1McPzrd/x6.png" alt="App Screenshot" width="600">
</p>


  b) PlotFeatures: Displays the predicted class for the input image
  <p align="center">
  <img src="https://i.postimg.cc/G3QZK6Ry/x7.png" alt="App Screenshot" width="180">
</p>



## Tech Stack

- **Language**: MATLAB
- **Libraries**: Deep Learning Toolbox, Image Processing Toolbox
- **Dataset**: MNIST (Handwritten Digit Dataset)
- **Architecture**: CNN with Convolution, ReLU, Pooling, and Softmax layers

## Installation

1. Run `TestMnistConv()` to execute the CNN for digit recognition.
2. Check the generated 'MnistConv.mat' file for the results.
3. Visualize the convolution, ReLU, pooling, and softmax operations using `PlotFeatures()`.
