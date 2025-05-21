# NaanMudhalavan
📁 Project 1
Brain Tumor Segmentation using OpenCV and Deep Learning

📝 Overview
A computer vision and deep learning-based solution for automated brain tumor segmentation from MRI scans, aiming to assist in medical diagnosis, surgical planning, and telemedicine applications. This project implements both classic and deep learning approaches (e.g., U-Net), coupled with a clean and interactive Streamlit front-end.

🚀 Key Features
Preprocessing of MRI and mask images.

Segmentation using OpenCV techniques (thresholding, edge detection).

Deep Learning models (U-Net) for accurate tumor boundary detection.

Performance evaluation using Dice Coefficient, IoU, Precision, and Recall.

Streamlit-based tool for image upload, visualization, and result download.

🧠 Business Use Cases
Clinical Diagnosis: Support radiologists with high-precision segmentations.

Telemedicine: Enable remote diagnosis in rural areas.

Medical Education: Train students with real MRI data.

Surgical Planning: Provide tumor contours to neurosurgeons.

Research: Facilitate oncology studies and drug development.

📊 Model Evaluation Metrics
Dice Coefficient: 0.80–0.90

IoU: 0.75–0.85

Inference Time: <1 sec per image

Generalization to unseen data

🛠️ Tech Stack
Python, OpenCV, TensorFlow/Keras or PyTorch, Streamlit, Matplotlib, Scikit-learn

📦 Deliverables
🧠 Trained Segmentation Model

📓 Jupyter Notebook with full workflow

🌐 Streamlit App

📁 Project 2

🛡 Military Soldier Safety and Weapon Detection
📌 Project Overview
This project focuses on enhancing soldier safety and battlefield awareness by using Computer Vision and YOLO object detection to identify threats such as weapons, enemy combatants, and unauthorized vehicles in real-time. A Streamlit-based web app is developed to make the system accessible, visual, and interactive.

🎯 Objectives
Detect weapons, soldiers, vehicles, and trenches using visual data.

Distinguish between friendly forces, enemies, and civilians.

Provide real-time alerts and visual feedback.

Improve situational awareness for command and field units.

Operate effectively in urban, forest, and desert environments.

🧠 Skills Gained
Computer Vision & Image Processing

Deep Learning with YOLO

Real-time Object Detection

Exploratory Data Analysis (EDA)

Streamlit-based Web App Development

🔍 Key Features
📸 Upload and analyze images/videos

🧠 Automatic object detection (soldiers, weapons, vehicles, civilians, trenches)

🚨 Threat classification: friendly vs enemy vs civilian

📊 Visualizations: bounding boxes, heatmaps, class distribution

💾 Download detection results for further use

🌐 Web interface for ease of use

🧰 Technologies Used
YOLOv5 – for object detection

OpenCV – for image processing

Streamlit – for frontend interface

Pandas, Matplotlib, Seaborn – for data visualization

Optional OCR – Tesseract / EasyOCR

🧾 Classes Detected
Class ID	Object
0	Camouflage Soldier
1	Weapon
2	Military Tank
3	Military Truck
4	Military Vehicle
5	Civilian
6	Soldier
7	Civilian Vehicle
8	Trench

📊 Performance Summary
mAP (mean Average Precision): 85%

Precision: 88%

Recall: 83%

Threat Classification Accuracy: 92%

Inference Speed:

30 FPS (GPU)

10 FPS (Edge Devices like Jetson Nano)

📁 Dataset
Link: Click to access dataset

Contains annotated images of military/civilian scenarios

Follows YOLO format (class, x_center, y_center, width, height)

🔬 Exploratory Data Analysis Highlights
Image quality, size, aspect ratio consistency

Bounding box size and class distribution

Heatmaps for object density

Class imbalance solutions (augmentation/synthetic data)

Train-test-validation balance checked

📈 Evaluation Metrics
Precision – Measures how accurate the detections are

Recall – Measures how many actual threats are detected

mAP – Aggregated detection accuracy

F1 Score – Balance between precision and recall

Inference Time – Speed of real-time detection

📦 Project Deliverables
Streamlit Web App

Model Performance Report

Documentation with Results & Visuals

Jupyter Notebook (optional)

Sample Detection Outputs

EDA Visualizations

🕒 Timeline
The project is structured for a 10-day completion. Support is available via:

Doubt Sessions: Tue, Thu, Sat (5:00PM – 7:00PM)
Book a Session

Live Evaluations: Mon – Sat (11:30PM – 12:30PM)
Register Here

✅ Key Achievements
Accurate real-time detection in battlefield conditions

High threat classification accuracy

Streamlined interface for deployment and monitoring

Operates across diverse environmental conditions



📁 Project 3
Number Plate Detection using YOLO and OCR

📝 Overview
This project implements a real-time Number Plate Detection System using the YOLO object detection framework and OCR (Tesseract/EasyOCR). It supports both image uploads and webcam feeds, enabling the automatic detection and extraction of license plate text.

🚀 Key Features
YOLO-based real-time number plate detection.

OCR integration for accurate alphanumeric text extraction.

Robust to lighting, occlusion, and multiple plate formats.

Streamlit web interface with upload, webcam, and download options.

Performance dashboard (detection time, accuracy, etc.).

🧠 Business Use Cases
Traffic Management: Automated fines and vehicle tracking.

Toll Collection: Seamless toll booth access.

Law Enforcement: Detect stolen or unauthorized vehicles.

Parking & Access Control: Restrict unauthorized vehicles.

Fleet Monitoring: Real-time logistics tracking.

📊 Model Evaluation Metrics
Mean Average Precision (mAP)

OCR Accuracy

Inference Time

False Positive/Negative Rates

🛠️ Tech Stack
Python, OpenCV, YOLO, Streamlit, Tesseract/EasyOCR, Pandas, NumPy, Matplotlib

📦 Deliverables
✅ Trained YOLO Model

🗂️ Augmented Dataset

🌐 Streamlit Web App

📄 OCR Outputs & Reports

🎥 Demo Video

🧪 Evaluation Metrics

📁 Project 4
Damaged Car Image Preprocessing Pipeline

📝 Overview
This project implements a complete image preprocessing pipeline tailored for damaged car images using OpenCV and Streamlit. The goal is to automate preprocessing tasks such as resizing, cropping, thresholding, and color space conversion to assist in downstream applications like damage detection, repair estimation, and insurance claim automation.

🚀 Key Features
Resize and crop damaged areas for enhanced analysis.

Color space conversion: RGB, BGR, HSV.

Adaptive thresholding techniques.

Streamlit web app for interactive image processing.

Downloadable preprocessed outputs.

Exploratory Data Analysis (EDA) to understand image dimensions and damage patterns.

🧠 Business Use Cases
Insurance Claims: Faster claim processing using clean images.

Automotive Repairs: Support automated repair cost estimation.

Manufacturing QA: Identify vehicle defects at production stages.

AI Training: Provide clean input for damage detection models.

🛠️ Tech Stack
Python, OpenCV, Streamlit, EDA, Image Processing

📦 Deliverables
📓 Jupyter Notebook

🌐 Streamlit App

📁 Preprocessed Image Outputs

📄 Project Documentation
