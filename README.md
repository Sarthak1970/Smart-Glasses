# Smart-Glasses
This repository contains the code and resources for developing smart glasses aimed at assisting visually impaired individuals by providing features such as object detection, text recognition, and environmental awareness. The project leverages computer vision and AI to make the world more accessible.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Hardware Requirements](#hardware-requirements)
- [Architecture](#architecture)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
Smart Glasses for the Visually Impaired are designed to assist people with visual impairments by using AI-powered features to interpret and describe their surroundings. The glasses use a combination of cameras, sensors, and display feedback to relay important information to the user.

## Features
- **Object Detection**: Identifies objects in the environment and provides audio feedback to describe them.
- **Text Recognition (OCR)**: Reads printed or handwritten text and converts it to speech.
- **Navigation Assistance**: Alerts the user to obstacles and provides directions.
- **Face Recognition**: Identifies known faces and notifies the user.
  

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/smart-glasses.git
   cd smart-glasses
   ```
2. **Install Dependencies**:
   - Make sure to have Python installed. Then, use `pip` to install the necessary packages:
     ```bash
     pip install -r requirements.txt
     ```
3. **Set Up Hardware**:
   - Connect the Pi-Cam module and display output device to the processing unit.

## Usage
1. **Run the Main Program**:
   ```bash
   python main.py
   ```
2. The glasses will start processing video input, detect objects, recognize text, and provide audio feedback.
3. Adjust settings as needed in the `config.json` file to optimize performance for your environment.

## Hardware Requirements
- **Processing Unit**: Raspberry Pi 5(4gb)

  
  ![image](https://github.com/user-attachments/assets/4a72fe96-931b-4b2c-8800-62a91e09237e)

- **Camera Module**:  Pi-Cam for image capture and streaming
  ![image](https://github.com/user-attachments/assets/43f43a63-486a-4862-bc6c-4322246fa5d9)


  
- **Oled Display**: shows retrieved images/text

  
  ![image](https://github.com/user-attachments/assets/b050d4a6-117e-4261-9fc5-e5e2a9b4f814)

  
- **Battery Pack**: Portable power supply for mobility

## Architecture
The smart glasses system is composed of several components:
1. **Camera Input**: Captures real-time video and images using the Pi-Cam module.
2. **AI Processing Unit**: Analyzes visual data using models for object detection, OCR, Image captioning and facial recognition.
3. **Audio Feedback**: Converts visual information into speech for the user.
4. **Navigation Assistance**: Uses sensors to detect obstacles and guide the user.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
