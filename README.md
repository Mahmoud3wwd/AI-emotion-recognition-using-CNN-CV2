
# Emotion Recognition Using CNN

This project recognizes human emotions from facial expressions using Convolutional Neural Networks (CNNs). The emotions categorized are:
- Sad
- Angry
- Happy
- Neutral
- Fear

The model is trained on a dataset of labeled facial images. The trained model is integrated with OpenCV to capture live video frames from a camera feed, detect faces, and predict emotions in real-time.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Files](#files)
6. [Training](#training)
7. [Notes](#notes)
8. [Future Improvements](#future-improvements)

## Overview

This project aims to recognize human emotions from facial expressions using Convolutional Neural Networks (CNNs). The emotion categories include: sad, angry, happy, neutral, and fear.

## Requirements

- Python 3.x
- OpenCV (cv2)
- TensorFlow
- Keras
- NumPy

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/mahmoud3wwd/AI-emotion-recognition-using-CNN-CV2.git
   cd AI-emotion-recognition-using-CNN-CV2

## Usage
python main.py 
- This script will open a live video stream from your webcam and overlay the predicted emotion on detected faces.


## Files

- {main.py} : Main script for video capture, face detection, and emotion recognition.
- {emotionai.py} : Jupyter notebook for training the CNN model.

##Training
- Model trained using emotionai.py.
- Dataset used: [(https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)]
- Model architecture: [CNN architecture details]
- Training parameters: [Epochs, batch size, optimizer, etc.]


##Notes
- face detection using Haar cascades (haarcascade_frontalface_default.xml) from OpenCV.
- Ensure proper lighting and camera positioning for accurate emotion recognition.- Model architecture: [CNN architecture details]
- Assumes webcam connected to system.


##Future improvment
- Implement robust face detector (e.g., Dlib, MTCNN) for better performance..
- Explore ensemble methods for emotion classification.
- rain on larger dataset for improved accuracy.
   


![image](https://github.com/Mahmoud3wwd/AI-emotion-recognition-using-CNN-CV2/assets/150680874/4906b218-eee5-45fc-8b1a-41a978ac3021)
![image](https://github.com/Mahmoud3wwd/AI-emotion-recognition-using-CNN-CV2/assets/150680874/dd6d552a-2e60-457b-a5d3-0971c521c871)



