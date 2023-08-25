# Real-time Emotion Detection and Chatbot 

Welcome to the repository containing Python code for a real-time emotion detection system and a chatbot web application. This project combines computer vision and natural language processing (NLP) techniques to detect emotions from webcam video streams and engage in interactive text-based conversations.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Installation](#installation)
- [Example](#example)
- [Contributing](#contributing)


## Introduction
This repository demonstrates the integration of multiple technologies, including real-time emotion detection using computer vision and a chatbot powered by OpenAI's GPT-3 model. The web application is built using Flask, a lightweight web framework.

## Features
- **Real-time Emotion Detection:** The code uses a pre-trained convolutional neural network (CNN) model to detect emotions from live webcam video streams. It identifies faces in the video, extracts regions of interest (ROI), and classifies the emotions as "Angry," "Happy," "Sad," and more.
- **Chatbot Interaction:** The web application includes a chatbot that engages in text-based conversations with users. The chatbot leverages OpenAI's GPT-3 model to generate human-like responses.
- **Webcam Integration:** The application captures video streams from the user's webcam and overlays emotion labels on detected faces in real-time.
- **User Interface:** The application provides a simple web interface where users can view live webcam feeds and chat with the chatbot.

## Usage
1. Clone or download the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Open a web browser and navigate to `http://localhost:5000` to access the application.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/emotion-detection-chatbot.git
   ```
2. Navigate to the project directory:
   ```sh
   cd emotion-detection-chatbot
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```sh
   python app.py
   ```

## Example
1. Open a web browser and go to `http://localhost:5000`.
2. Navigate to the webcam view to see real-time emotion detection on your face.
3. Access the chatbot page and engage in text-based conversations with the chatbot.

## Contributing
Contributions are welcome! Feel free to enhance the features or fix any issues by creating a pull request.

*Note: This README provides an overview of the code's functionality and usage. For detailed implementation, please refer to the source code in `mainn.py`.*
