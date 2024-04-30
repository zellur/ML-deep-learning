# ML-deep-learning
Deep learning models


Malaria Detection using Convolutional Neural Networks

This project utilizes a Convolutional Neural Network (CNN) model to detect Malaria from cell images. The model is built using TensorFlow and Keras libraries in Python.  

Features: 

The project uses the Malaria dataset from TensorFlow Datasets.
The dataset is split into training, validation, and testing sets.
The images in the dataset are preprocessed and resized for the model.
The model architecture is based on the LeNet-5 architecture, with added Batch Normalization layers for improved performance.
The model is trained and validated on the respective datasets, and the performance is evaluated on the test set.
The training and validation loss and accuracy are plotted for each epoch to visualize the model's learning process.
The trained model can predict whether a cell image is infected with Malaria or not.


Usage:

To run the project, execute the malaria_detection.py script. The script will train the model and evaluate its performance on the test set. The model's prediction for a sample image from the test set is also displayed.  
Requirements
Python -> 3
TensorFlow -> 2.15
TensorFlow Datasets
Matplotlib
