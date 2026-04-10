# OCR-From-Scratch-NumPy-MLP-MNIST-Pygame-UI-
Handwritten digit recogniser built from scratch using NumPy. Includes manual backpropagation, MNIST parsing, occlusion training, and an interactive Pygame drawing interface.

# OCR From Scratch (NumPy MLP + MNIST + Pygame UI)

A handwritten digit recognition system built entirely from first principles using NumPy, trained on MNIST, and deployed in an interactive Pygame drawing interface.

This project implements a full machine learning pipeline without using deep learning libraries such as PyTorch or TensorFlow. Everything from data loading to backpropagation is written manually.

---

## Description

Handwritten digit recogniser built from scratch using NumPy. Includes manual backpropagation, MNIST parsing, occlusion-based training, and a real-time drawing interface using Pygame.

---

## Overview

The model is a simple 2-layer neural network (MLP) trained on 28x28 grayscale images from the MNIST dataset.

To make the task more realistic, the training data is augmented using random occlusion. Parts of digits are masked during training so the model learns to handle incomplete or noisy inputs instead of only clean ones.

After training, a Pygame interface allows the user to draw digits and get predictions instantly.

This was one of my first attempts at building a neural network from scratch. I avoided using ML frameworks to better understand how everything works internally. Some inspiration also came from reading "The New Turing Omnibus", which encouraged building systems from first principles.

---

## Features

- MNIST dataset downloaded and parsed manually from IDX files
- Fully connected neural network implemented in NumPy
- Manual implementation of:
  - forward propagation
  - ReLU activation
  - softmax output
  - cross-entropy loss
  - backpropagation
- Mini-batch stochastic gradient descent
- Random occlusion data augmentation during training
- Save and load model weights
- Interactive Pygame drawing interface
- Optional quiz mode to track prediction accuracy manually

---

## Model Architecture

Input layer:
- 784 features (flattened 28x28 image)

Hidden layer:
- 256 units
- ReLU activation

Output layer:
- 10 units (digits 0–9)
- softmax activation

---

## Training Details

- Loss function: cross-entropy
- Optimiser: stochastic gradient descent (manual)
- Batch size: 128 (default)
- Learning rate: 0.01 (default)
- Dataset: MNIST (subset used by default for speed)

### Data Augmentation

During training, around 60% of images are randomly occluded:
- a small rectangular region is set to zero
- patch size varies between 8% and 20% of the image

This forces the model to learn more robust features rather than memorising clean digit shapes.

---

## Results

Typical performance with default settings:
- Training accuracy: ~97–99%
- Test accuracy: ~94–96%

Accuracy on user-drawn digits is lower due to differences between real drawings and MNIST data.

---

## How It Works

Forward pass:
z1 = XW1 + b1  
h1 = ReLU(z1)  
z2 = h1W2 + b2  

Loss:
Cross-entropy applied to softmax probabilities

Backpropagation:
Gradients are computed manually and applied using SGD updates.

---

## GUI (Pygame)

The interface allows:
- drawing digits on a canvas
- clicking predict to classify the digit
- clearing the canvas
- optional quiz mode to track correctness

Input pipeline:
- canvas converted to grayscale
- resized to 28x28
- flattened to 784 vector
- passed into the model

---

## Installation

pip install numpy pygame

---

## Usage

Train the model:
python ocr_partially.py --train

Launch GUI:
python ocr_partially.py --gui

Launch GUI with quiz mode:
python ocr_partially.py --gui --quiz

Custom training example:
python ocr_partially.py --train --epochs 5 --batch 128 --lr 0.01 --subset 20000

Delete saved weights:
python ocr_partially.py --clear

---

## Project Structure

ocr_partially.py  
partial_ocr_weights.npz  
data/  
  train-images-idx3-ubyte.gz  
  train-labels-idx1-ubyte.gz  
  t10k-images-idx3-ubyte.gz  
  t10k-labels-idx1-ubyte.gz  

---

## Limitations

- Training in NumPy is slow compared to GPU-based frameworks
- Model is shallow and does not use convolutional layers
- User drawings differ from MNIST distribution
- Minimal preprocessing on drawn input

---

## Possible Improvements

- Add convolutional neural network (CNN)
- Improve preprocessing (centering and scaling digits)
- Visualise training curves
- Add confusion matrix evaluation
- Compare occlusion vs non-occlusion training
- Deploy as a web app

---

## Summary

This project demonstrates:
- understanding of neural network fundamentals
- ability to implement backpropagation from scratch
- experience working with raw dataset formats
- ability to connect machine learning to an interactive application
