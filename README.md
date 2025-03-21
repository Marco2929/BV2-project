## 🚦 Traffic Sign Recognition with YOLOv8  
**Computer Vision 2 – Heilbronn University (Summer Semester 2024)**  
**Marco Menner & Benedikt Seeger**  
Degree Program: Mechatronics and Robotics  

---

### 📌 Project Overview

As part of the course **Computer Vision 2**, we developed an AI-based system for **real-time traffic sign recognition and classification**. The goal was to detect various types of traffic signs reliably — even under challenging real-world conditions such as poor lighting, occlusions, and perspective distortion.

The system is based on **YOLOv8n**, a lightweight real-time object detection model, and uses modern techniques such as **mosaic augmentation**, **transfer learning**, and a **two-stage classifier** specifically for speed limit signs.

---

### 🧰 Technologies & Methods
- **YOLOv8n** (object detection + classification)
- **Custom synthetic dataset** with random backgrounds and augmentations
- **Negative sampling** to reduce false positives
- **Frame-based caching** to stabilize detection
- **Python, OpenCV, Ultralytics YOLO**

---

### 🎯 Key Features
- Real-time traffic sign detection with high accuracy
- F1-Score > 98% at confidence threshold 0.81
- Visual overlay interface with a persistent speed sign display
- Two-stage classification for precise speed limit recognition
- Modular architecture with flexible dataset handling and training scripts

---

### 📈 Training & Evaluation
- Data augmentation using blur, distortion, HSV shift, and perspective transforms
- Trained on 3000+ custom-generated images
- Evaluated using real-world video sequences and GTSDB benchmark
- Performance metrics: confusion matrix, F1-score curves, visual detections

---

### ⚙️ Inference Pipeline
- Detection time per frame: **0.06 – 0.09 seconds**
- Speed signs are highlighted and persist until replaced
- Other signs are displayed in a rotating "sign box" for the last 6 detections
- Detection is confirmed only after consistent recognition over 3 frames

---

### 🚀 Future Improvements
- Add more sign types (e.g., temporary signs, LED-based signs)
- Integrate traffic light state detection
- Implement region-of-interest filtering for driver-relevant signs
- Port detection/classification to **C++** for embedded real-time deployment

---

Would you like a full `README.md` including install/setup instructions, usage commands, and maybe a screenshot section? Happy to prep that for you!
