DATA SET : https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

Emotion Recognition Using CNN
Overview
This project aims to recognize human emotions from facial expressions using Convolutional Neural Networks (CNNs). The emotion categories include:

Sad
Angry
Happy
Neutral
Fear
The model is trained on a dataset of facial images labeled with these emotions. The trained model is then integrated with OpenCV to capture live video frames from a camera feed, detect faces, and predict the emotions of the detected faces in real-time.

Requirements
Python 3.x
OpenCV (cv2)
TensorFlow
Keras
NumPy
Setup
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/emotion-recognition-cnn.git
cd emotion-recognition-cnn
Install the required Python packages:

Copy code
pip install -r requirements.txt
Download the pre-trained CNN model weights (if provided separately).

Usage
Run emotion_recognition.py to start the emotion recognition system:
Copy code
python emotion_recognition.py
This script will open a live video stream from your webcam and overlay the predicted emotion on the detected faces.
Files
emotion_recognition.py: Main script to capture video, detect faces, and recognize emotions.
train_emotion_model.ipynb: Jupyter notebook for training the CNN model.
model_weights.h5: Pre-trained model weights.
haarcascade_frontalface_default.xml: XML file for face detection using OpenCV's Haar cascades.
Training
The model was trained using the train_emotion_model.ipynb notebook.
Dataset used: [Dataset Name/Source]
Model architecture: [CNN architecture details]
Training parameters: [Epochs, batch size, optimizer, etc.]
Notes
The face detection is performed using Haar cascades (haarcascade_frontalface_default.xml) from OpenCV.
Ensure proper lighting and positioning of the camera for accurate emotion recognition.
This project assumes a webcam is connected to the system.
Future Improvements
Implement a more robust face detector (e.g., using Dlib or MTCNN) for better performance.
Explore ensemble methods for emotion classification.
Train on a larger dataset for improved accuracy and generalization.

![image](https://github.com/Mahmoud3wwd/AI-emotion-recognition-using-CNN-CV2/assets/150680874/4906b218-eee5-45fc-8b1a-41a978ac3021)
![image](https://github.com/Mahmoud3wwd/AI-emotion-recognition-using-CNN-CV2/assets/150680874/dd6d552a-2e60-457b-a5d3-0971c521c871)



