# Employee Wellbeing through Emotion Detection (v2)

## Overview
The **Real-Time Emotion Monitoring and Analytics Dashboard** is a cutting-edge tool designed to enhance employee well-being by analyzing facial expressions in real-time using deep learning. This project leverages CUDA-accelerated computation on an NVIDIA RTX 4060 GPU with CUDA 12.6, enabling rapid training and inference of emotion detection models. It surpasses traditional surveys by providing dynamic, immediate insights into workforce morale. The system processes 2-minute video feeds, detects emotions with high accuracy, and stores results in PostgreSQL with split `date_stamp` and `time_stamp` fields—all while ensuring ethical use through consent and privacy via data deletion. A FastAPI endpoint delivers this data to power interactive frontend dashboards, helping organizations foster a healthier workplace.

This is **version 2 (v2)** of the project, featuring significant upgrades over v1, including a fully interactive frontend and enhanced user experience.

## What's New in v2
- **Fully Interactive Frontend with TailwindCSS + HTML**:
  - A modern, responsive homepage (`dashboard.html`) built with [TailwindCSS](https://tailwindcss.com/) and HTML, offering an engaging entry point to the system.
  - A single-page application (SPA) interface (`index.html`) with seamless navigation across login, model selection, real-time detection, and admin dashboard views.
  - Real-time updates and live video streaming powered by [SocketIO](https://socket.io/), delivering immediate feedback during emotion detection.
  - Interactive visualizations in the admin dashboard using [ApexCharts](https://apexcharts.com/) to showcase emotion trends and workforce insights.

- **CUDA-Optimized Emotion Detection**:
  - Leverages the NVIDIA RTX 4060 GPU with CUDA 12.6 for fast, efficient training and real-time inference of emotion detection models.
  - Processes 2-minute video feeds with low-latency emotion detection, ensuring smooth performance during live sessions.

## Features
- **CUDA-Accelerated Model Training:**
  - Trains ResNet50 (pretrained, adapted for grayscale) ,Efficient Net and a  custom CNN on the FER dataset for 20 epochs each.
  - Uses PyTorch with mixed precision training (`torch.amp`) and CUDA for maximum GPU performance.
  - Generates visualizations (loss/accuracy plots) with progress bars (`tqdm`) for real-time feedback.
![WhatsApp Image 2025-07-09 at 21 43 18_c47b3fe9](https://github.com/user-attachments/assets/d3be6b8f-9b25-4f38-a088-1189e48bc7af)


- **Model Testing with GUI:**
  - Validates models on test images using CUDA for fast inference.
  - Displays results in a Tkinter GUI popup with predicted emotions (e.g., happy, sad).


- **Real-Time Emotion Detection:**
  - Captures 2-minute webcam video, processing frames on the GPU with CUDA for low-latency detection using ResNet50.
  - Deletes temporary files post-processing for privacy, limiting detection to 2 minutes for efficiency.
  - Stamps frames with date/time for tracking.

![WhatsApp Image 2025-07-07 at 20 35 04_90ae9fc8](https://github.com/user-attachments/assets/937e6ab6-d23f-41ba-8892-afdf2f548fb7)


- **Database Storage:**
  - Stores results in PostgreSQL with split `date_stamp` (e.g., `2025-03-18`) and `time_stamp` (e.g., `14:30:45`).
  - Records user ID, department, and emotion (0-6) per frame, enforcing consent via GUI.
![WhatsApp Image 2025-07-09 at 21 42 50_c81aaab3](https://github.com/user-attachments/assets/c0215a70-7651-4038-a9a2-9c439c166e51)

- **API for Frontend Integration:**
  - Provides a FastAPI endpoint (`/emotions`) to fetch emotion data as JSON for dashboard integration.
 
## Output 

![WhatsApp Image 2025-07-07 at 20 06 47_210977fb](https://github.com/user-attachments/assets/62e66e9a-ac33-41a9-9f98-49162935f4f2)
![WhatsApp Image 2025-07-07 at 20 07 15_e9b87494](https://github.com/user-attachments/assets/23d3f03f-b2af-4bc3-b8ce-2782a3f0adb9)
![WhatsApp Image 2025-07-07 at 20 07 48_3e073e87](https://github.com/user-attachments/assets/d4621840-e079-4c88-8c3e-c96b1b48d5f2)
![WhatsApp Image 2025-07-07 at 20 08 04_6948e4a1](https://github.com/user-attachments/assets/cd5178f8-5c64-4767-bb58-96f85ac64d3b)
![WhatsApp Image 2025-07-07 at 20 08 24_fcdf9d3e](https://github.com/user-attachments/assets/22cadc2f-ad41-46ef-9381-eb4e830e17f4)
![WhatsApp Image 2025-07-07 at 20 08 42_45a005d3](https://github.com/user-attachments/assets/ebc76dd3-f9b1-4e81-9592-45b9c36ed1fb)
![WhatsApp Image 2025-07-07 at 20 35 04_3620beba](https://github.com/user-attachments/assets/ea494970-8f29-4ed7-be89-1789e877cfa7)
![WhatsApp Image 2025-07-07 at 20 35 34_fd5a3443](https://github.com/user-attachments/assets/24a2ea49-c5c5-479b-8735-29bd4212b2b5)
![WhatsApp Image 2025-07-07 at 20 36 07_383e0093](https://github.com/user-attachments/assets/169bdf83-f704-45a2-bd24-bf31edb502dc)

##Admin Dashboard:

![WhatsApp Image 2025-07-07 at 20 37 18_5ee7b4ba](https://github.com/user-attachments/assets/832ac41a-fc96-4f25-86bc-4fe9d4a130df)


## Directory Structure

```
Employee-Wellbeing-Emotion-Detection/
├── data/                  # FER dataset (not included in repo)
│   ├── train/            # Training images b   y emotion
│   └── test/             # Testing images by emotion
├── saved_emotion_model/   # Model save directory (empty in repo)
│   ├── Assets/           # Training visualizations
│   └── Variables/        # Trained model weights (.pth)
├── static/                # Static files for frontend
│   ├── office1.png       # Image for dashboard
│   ├── office2.png       # Image for dashboard
│   ├── styles.css        # Custom CSS (if any)
│   ├── main.js           # JavaScript for frontend logic
│   └── apexcharts.min.js # ApexCharts library
├── dashboard.html         # Public homepage (TailwindCSS + HTML)
├── index.html             # Main application interface (SPA with TailwindCSS)
├── train_models.py        # CUDA-accelerated training script
├── test_model.py          # Testing script with GUI
├── real_time_detection.py # Real-time detection with database storage
├── database_setup.sql     # PostgreSQL schema
├── api_server.py          # FastAPI server
├── README.md              # Project docs
├── requirements.txt       # Dependencies
└── .gitignore             # Excluded files
```

## Tech Stack
- **Programming Language:** Python 3.9+
- **Deep Learning:**
  - **PyTorch:** Core framework with CUDA support for GPU-accelerated tasks.
  - **Torchvision:** Supplies pretrained models and data utilities, optimized for CUDA.
- **Computer Vision:** OpenCV (`opencv-python`) for video capture and frame processing.
- **GUI:** Tkinter for user interfaces.
- **Database:** PostgreSQL for structured data storage.
- **API:** FastAPI with Uvicorn for high-performance API serving.
- **Frontend:**
  - **TailwindCSS:** Utility-first CSS framework for responsive, modern interfaces.
  - **HTML:** Structures the interactive frontend pages.
  - **JavaScript:** Powers SPA logic and real-time features.
  - **ApexCharts:** Interactive charting for data visualization.
  - **SocketIO:** Real-time communication for live updates.
- **Utilities:**
  - **Matplotlib:** Plots training metrics.
  - **Tqdm:** Progress bars for training.
  - **Pillow (PIL):** Image handling.
  - **Psycopg2:** PostgreSQL connectivity.
- **Hardware:** NVIDIA RTX 4060 GPU with CUDA 12.6 and cuDNN.

## Requirements
Install via `requirements.txt`:

## Setup
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Setup PostgreSQL:**
   - `CREATE DATABASE emotion_db;`
   - Run `database_setup.sql`.
3. **Update Credentials:** Replace `your_password` in `real_time_detection.py` and `api_server.py`.
4. **Prepare Dataset:** Place FER dataset in `data/train/` and `data/test/`.
5. **Run Training:** `python train_models.py` (CUDA-accelerated).
6. **Test Model:** `python test_model.py` (CUDA inference).
7. **Real-Time Detection:** `python real_time_detection.py` (CUDA processing).
8. **Start API:** `uvicorn api_server:app --reload` (access at `http://127.0.0.1:8000/emotions`).

## Dataset
- **FER Dataset:** From [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013), not included due to size.

## Hardware
- Optimized for NVIDIA RTX 4060 with CUDA 12.6 and cuDNN on Windows.

## Version
`v2` as of July 09, 2025.
