
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

1. **Load the MNIST Dataset**:
- Download the MNIST dataset files (t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte) and place them in the project directory.

2. **Run the main script**: 
- Open MATLAB and run: matlab and run TestMnistConv.m (for epochs and accuracy) and PlotFeatures.m (for the predicted class)




## Running Tests

The project can be implemented and tested to verify funtionality

