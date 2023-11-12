
# Flower Image Classifier Project

## My Journey

Welcome to my Flower Image Classifier project! This venture was part of my Udacity capstone experience, showcasing my journey in training a neural network to classify images of flowers into 102 different categories.

## What I Built

### `train.py`

In this script, I poured my efforts into training a new neural network. I opted for the VGG16 architecture, a powerful choice for image classification. The training spanned 10 epochs, with a batch size of 64 for training and 32 for both validation and testing.

### `predict.py`

Once the model was trained, I crafted a script to predict the classes of new images. This script loads the checkpoint saved during training and uses the model to predict the class of a given image.

### `checkpoint.pth`

This file holds the heart of my project—the saved state of the trained model. It encapsulates the model's architecture, weights, the mapping of classes to indices, and other crucial details.

## Dependencies
Make sure to have the following dependencies installed:

* Python
* PyTorch
* NumPy
* Matplotlib
* PIL (Pillow)

## Navigating Challenges

Building this project wasn't all smooth sailing. Here are some challenges I encountered and how I tackled them:

### GPU Memory Hiccups

GPU memory errors gave me a headache. I learned to tread carefully, using `torch.cuda.empty_cache()` and adjusting memory allocations when necessary.

### PyTorch Version Juggling

Consistency in PyTorch versions across different platforms is key. I made sure to document version requirements to avoid unexpected hiccups.

### Time Crunch

Training deep neural networks takes time. I explored optimization strategies, delved into parallelization, and even considered cloud-based GPU resources for a quicker turnaround.

### Debugging Odyssey

Implementing robust error handling became my North Star. Print statements and detailed logging were my allies in catching and resolving issues early in the development process.

## Results and Reflections

After 10 epochs, my model achieved an impressive accuracy of 85% or more on the validation set. The journey was enriching, filled with experimentation and learning.

## Shoutouts

A huge shoutout to Udacity for providing the project framework and guidance. Also, immense gratitude to the PyTorch and open-source communities for contributing to the success of this project.

Here's to classifying flowers with the model I crafted! 🌸🌺🌼
