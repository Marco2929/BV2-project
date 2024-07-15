Here is the revised README with improved grammatical correctness:

# BV2-project

This project includes a Python script capable of detecting traffic signs in a video and overlaying additional
information. Additionally, the project provides scripts for dataset generation and training.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Overview](#project-overview)

## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed Python 3.10. You can download it from [python.org](https://www.python.org/).
- You have a working internet connection.

### Steps

1. **Navigate to the project directory:**

   Open your terminal or command prompt and navigate to the directory where your project is located:
    ```bash
    cd BV2-project
    ```

2. **Create a virtual environment:**

   It's a good practice to create a virtual environment for your project. Run the following commands:
    ```bash
    python -m venv venv
    ```
   Activate the virtual environment:
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3. **Install the required packages:**

   Install the packages listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Project Overview

This section gives you a quick overview of the different folders.

### Data

The [data](data) folder contains all datasets used for training and evaluating the model. It also contains the videos
used for evaluation and the inference demo of the model. Furthermore, it includes our generated augmented dataset.
In the [augmented_dataset](data/augmented_dataset) folder, the final generated dataset is located.
The [background](data/background) folder contains the images from the COCO dataset, which are used as the background of
the generated dataset images.
The [basic_images](data/basic_images) folder contains the images of the traffic signs, which are randomly placed on one
of the background images.
The [gtsdb](data/gtsdb) folder contains the 600 images of this dataset, which are additionally used as test data.
The [speed_data](data/speed_data) folder contains all data used to train the speed sign classification model.
The [validation](data/validation) folder contains some validation images from the videos.
Finally, the [video](data/video) folder contains all videos used to test the model.
The [train.yaml](data/train.yaml) and [val.yaml](data/val.yaml) files contain the class list and the paths to the data
for the model.

### Demo_results

The [demo_results folder](demo_results) contains some of the videos with the results of the demo.

### Models

The [models](models) folder contains the final evaluated models used by the inference script. It contains the detection
model for overall traffic sign recognition and the speed sign classification model used in further processing.

### Results

The [results](results) folder contains the various results obtained during the training and validation process of the
model. This includes both the detection and classification models.

### Src

The [src](src) folder contains the actual running scripts for inference. The main script to run the traffic sign
recognition is [traffic_sign_recognition.py](/src/traffic_sign_recognition.py).
The frontend folder contains the display overlay script, which shows the detected sign and the corresponding images of
the sign in the image_utils folder.
The draw_bboxes script simply draws the bounding boxes in the image using OpenCV.

### Tools

The [tools](tools) folder contains various scripts to handle the datasets and bring them into an appropriate format. It
also includes our dataset generation script in the [generate_dataset](/tools/generate_dataset) folder.

### Training

The [training](training) folder contains the training and validation scripts for traffic sign detection and speed sign
classification. It also contains the pretrained yolov8n model file.

Note! The dataset currently just consists of 100 Train and Test images to keep the project small.

## Usage

This project includes multiple scripts. The following section will guide you through all the available functionalities.

### Dataset Generation

To generate a dataset by yourself, open the following [script](tools/generate_dataset/generate_dataset.py).
You can now change the specifications according to your needs. First, set the `number_of_dataset_images` to define how
many images will be created. You can also change the percentage of training images by adjusting the number (e.g., 0.7
for 70%).

### Training

The script to train YOLO is located [here](/training/train_yolo.py). To run it correctly, follow these steps:

1. Open the [train.yaml](data/train.yaml) and change the path variable to your system path where the augmented_dataset
   is located.
2. Now you can run the [train_yolo.py](/training/train_yolo.py) file.

### Inference

The script to run the inference demo on one of the videos is located [here](/src/traffic_sign_recognition.py).
The `video_path` variable contains the path to the video. Simply change the number according to the video you want to
use.