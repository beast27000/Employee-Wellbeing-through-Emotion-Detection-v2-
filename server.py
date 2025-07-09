from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import io
from PIL import Image
import torch
from torchvision import transforms
from arc import ModifiedResNet, EnhancedCNN, EnsembleModel
import time
import asyncio
import socketio
import logging
import os
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from typing import Optional
import re
import random

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI and SocketIO
app = FastAPI(exception_handlers={})  # Disable default exception handlers to avoid recursion
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*', logger=True, engineio_logger=False)
app_asgi = socketio.ASGIApp(sio)

# Add CORS middleware to handle OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/socket.io", app_asgi)  # Explicitly mount SocketIO to avoid conflicts

# Database connection
try:
    conn = psycopg2.connect(
        dbname="emotion_detection",
        user="postgres",
        password="Calcite*1234",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise

# Initialize placeholder employee
try:
    cursor.execute(
        "INSERT INTO employees (employee_id, department) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        ("unknown", "Unknown")
    )
    conn.commit()
except Exception as e:
    conn.rollback()
    logger.error(f"Failed to initialize placeholder employee: {e}")

# Add foreign key constraint to sessions table
try:
    cursor.execute("""
        ALTER TABLE sessions
        ADD CONSTRAINT fk_employee_id
        FOREIGN KEY (employee_id)
        REFERENCES employees (employee_id)
        ON DELETE RESTRICT;
    """)
    conn.commit()
except Exception as e:
    conn.rollback()
    logger.warning(f"Failed to add foreign key constraint: {e}")

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATHS = {
    "enhanced_cnn": "C:\\Employee Welbeing through Emotion Detection\\The Solution\\saved_emotion_model\\Variables\\enhanced_cnn_best.pth",
    "resnet18": "C:\\Employee Welbeing through Emotion Detection\\The Solution\\saved_emotion_model\\Variables\\resnet18_best.pth",
    "ensemble": "C:\\Employee Welbeing through Emotion Detection\\The Solution\\saved_emotion_model\\Variables\\ensemble_model.pth"
}
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
try:
    resnet = ModifiedResNet().to(device)
    cnn = EnhancedCNN().to(device)
    checkpoint = torch.load(MODEL_PATHS["ensemble"], weights_only=True)
    resnet.load_state_dict(checkpoint['resnet_state_dict'])
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    ensemble = EnsembleModel(resnet, cnn, checkpoint.get('weight1', 0.6), checkpoint.get('weight2', 0.4))
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    raise

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Track active camera sessions and user sessions
active_cameras = {}
user_sessions = {}  # Store employee_id and session_id per Socket.IO session

class LoginRequest(BaseModel):
    employee_id: str
    department: str

class SelectionRequest(BaseModel):
    model: str

class FeedbackLoginRequest(BaseModel):
    employee_id: Optional[str] = None
    department: Optional[str] = None
    password: Optional[str] = None

class FeedbackResponse(BaseModel):
    employee_id: str
    department: str
    question_1: int
    question_2: int
    question_3: int
    question_4: int
    question_5: int
    text_feedback: Optional[str]

def init_database(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback_logins (
                    employee_id VARCHAR(50) PRIMARY KEY,
                    department VARCHAR(50) NOT NULL,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback_responses (
                    id SERIAL PRIMARY KEY,
                    employee_id VARCHAR(50) REFERENCES feedback_logins(employee_id),
                    department VARCHAR(50) NOT NULL,
                    question_1 INT NOT NULL,
                    question_2 INT NOT NULL,
                    question_3 INT NOT NULL,
                    question_4 INT NOT NULL,
                    question_5 INT NOT NULL,
                    question_6 INT NOT NULL,
                    question_7 INT NOT NULL,
                    text_feedback TEXT,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        conn.rollback()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    logger.info("Root route accessed")
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Failed to read dashboard.html: {e}")
        raise HTTPException(status_code=500, detail="Error reading dashboard.html")

@app.get("/dashboard.html", response_class=HTMLResponse)
async def get_dashboard_html():
    logger.info("Dashboard route accessed")
    try:
        if os.path.exists("dashboard.html"):
            with open("dashboard.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            logger.warning("dashboard.html not found, serving index.html as fallback")
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Failed to read dashboard.html or index.html: {e}")
        raise HTTPException(status_code=500, detail="Error reading dashboard.html or index.html")

@app.get("/index.html", response_class=HTMLResponse)
async def get_index():
    logger.info("Index route accessed")
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Failed to read index.html: {e}")
        raise HTTPException(status_code=500, detail="Error reading index.html")

@app.get("/login-screen", response_class=HTMLResponse)
async def get_login_screen():
    logger.info("Login screen route accessed")
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Failed to read index.html for login-screen: {e}")
        raise HTTPException(status_code=500, detail="Error reading index.html")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.post("/login")
async def login(request: LoginRequest):
    logger.info(f"Login attempt with employee_id: {request.employee_id}, department: {request.department}")
    # Allow numeric or 3-letter employee_id
    if not (request.employee_id.isnumeric() or re.match(r'^[A-Za-z]{3}$', request.employee_id)):
        logger.error(f"Invalid employee_id format: {request.employee_id}")
        raise HTTPException(status_code=400, detail="Employee ID must be numeric or exactly 3 letters")
    if request.department not in ['IT', 'Accounting', 'Marketing', 'All']:
        logger.error(f"Invalid department: {request.department}")
        raise HTTPException(status_code=400, detail="Invalid department")
    try:
        cursor.execute("SELECT employee_id FROM employees WHERE employee_id = %s", (request.employee_id,))
        if not cursor.fetchone():
            logger.info(f"Employee ID {request.employee_id} not found, creating new entry")
            cursor.execute(
                "INSERT INTO employees (employee_id, department) VALUES (%s, %s)",
                (request.employee_id, request.department)
            )
            conn.commit()
        else:
            logger.info(f"Employee ID {request.employee_id} found in database")
        return {"message": "Login successful", "employee_id": request.employee_id}
    except Exception as e:
        conn.rollback()
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error during login: {str(e)}")

@app.post("/select")
async def select_model(request: SelectionRequest):
    if request.model not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Invalid model selected")
    try:
        global ensemble
        checkpoint = torch.load(MODEL_PATHS[request.model], weights_only=True)
        if request.model == "ensemble":
            resnet.load_state_dict(checkpoint['resnet_state_dict'])
            cnn.load_state_dict(checkpoint['cnn_state_dict'])
            ensemble = EnsembleModel(resnet, cnn, checkpoint.get('weight1', 0.6), checkpoint.get('weight2', 0.4))
        elif request.model == "resnet18":
            resnet.load_state_dict(checkpoint)
            ensemble = EnsembleModel(resnet, resnet, 1.0, 0.0)
        else:
            cnn.load_state_dict(checkpoint)
            ensemble = EnsembleModel(cnn, cnn, 1.0, 0.0)
        return {"message": "Model selected successfully"}
    except Exception as e:
        logger.error(f"Model selection error: {e}")
        raise HTTPException(status_code=500, detail="Error selecting model")

@app.post("/feedback_login")
async def feedback_login(request: FeedbackLoginRequest):
    try:
        if request.password:
            if request.password != "admin123":
                raise HTTPException(status_code=400, detail="Incorrect admin password")
            return {"message": "Admin login successful"}
        if not request.employee_id:
            raise HTTPException(status_code=400, detail="Employee ID is required for employee login")
        if not re.match(r'^[A-Za-z]{3}$', request.employee_id):
            raise HTTPException(status_code=400, detail="Employee ID must be exactly 3 letters")
        if not request.department or request.department not in ['IT', 'Accounting', 'Marketing', 'All']:
            raise HTTPException(status_code=400, detail="Invalid or missing department")
        cursor.execute(
            "INSERT INTO feedback_logins (employee_id, department, is_admin) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (request.employee_id, request.department, False)
        )
        conn.commit()
        return {"message": "Feedback login successful"}
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Database error in feedback_login: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in feedback_login: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/submit_feedback")
async def submit_feedback(request: FeedbackResponse):
    try:
        cursor.execute(
            """
            INSERT INTO feedback_responses 
            (employee_id, department, question_1, question_2, question_3, question_4, question_5, text_feedback)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                request.employee_id, request.department, request.question_1, request.question_2,
                request.question_3, request.question_4, request.question_5, request.text_feedback
            )
        )
        conn.commit()
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        conn.rollback()
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail="Error submitting feedback")

@app.get("/admin_feedback")
async def get_admin_feedback(employee_id: str):
    if not re.match(r'^[A-Za-z]{3}$', employee_id):
        raise HTTPException(status_code=400, detail="Employee ID must be 3 letters")
    try:
        cursor.execute(
            """
            SELECT department, question_1, question_2, question_3, question_4, question_5, text_feedback, submitted_at
            FROM feedback_responses WHERE employee_id = %s
            """,
            (employee_id,)
        )
        feedback = cursor.fetchall()
        if not feedback:
            raise HTTPException(status_code=404, detail="No feedback found for this employee")
        return [
            {
                "department": row[0],
                "question_1": row[1],
                "question_2": row[2],
                "question_3": row[3],
                "question_4": row[4],
                "question_5": row[5],
                "text_feedback": row[6],
                "submitted_at": row[7]
            } for row in feedback
        ]
    except Exception as e:
        logger.error(f"Admin feedback retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving feedback")

@sio.event
async def connect(sid, environ):
    try:
        logger.info(f"Socket.IO client connected: {sid}")
        active_cameras[sid] = {'active': False, 'cap': None, 'start_time': None, 'frame_count': 0, 'emotions_detected': {emotion: 0 for emotion in EMOTIONS}, 'emotion_history': []}
        user_sessions[sid] = {'employee_id': None, 'session_id': None}
        await sio.emit('connection_success', {'message': 'Connected to server'}, to=sid)
    except Exception as e:
        logger.error(f"Error in SocketIO connect event: {e}")
        await sio.emit('error', {'message': f'Connection error: {str(e)}'}, to=sid)
        raise

@sio.event
async def disconnect(sid):
    logger.info(f"Socket.IO client disconnected: {sid}")
    if sid in active_cameras and active_cameras[sid]['cap']:
        active_cameras[sid]['cap'].release()
    await save_session_data(sid)
    active_cameras.pop(sid, None)
    user_sessions.pop(sid, None)

@sio.event
async def set_employee_id(sid, data):
    try:
        logger.info(f"set_employee_id called with sid: {sid}, data: {data}")
        if sid in user_sessions and 'employee_id' in data:
            cursor.execute("SELECT employee_id FROM employees WHERE employee_id = %s", (data['employee_id'],))
            if cursor.fetchone():
                user_sessions[sid]['employee_id'] = data['employee_id']
                logger.info(f"Set employee_id {data['employee_id']} for sid {sid}")
            else:
                logger.warning(f"Invalid employee_id {data['employee_id']} for sid {sid}")
                await sio.emit('error', {'message': 'Invalid employee ID'}, to=sid)
        else:
            logger.error(f"Invalid set_employee_id call: sid {sid} not in user_sessions or employee_id missing")
            await sio.emit('error', {'message': 'Invalid employee ID data'}, to=sid)
    except Exception as e:
        logger.error(f"Error in set_employee_id: {e}")
        await sio.emit('error', {'message': f'Employee ID error: {str(e)}'}, to=sid)

async def save_session_data(sid):
    if sid not in active_cameras or active_cameras[sid]['frame_count'] == 0:
        return
    try:
        session_duration = time.time() - active_cameras[sid]['start_time']
        emotions_detected = active_cameras[sid]['emotions_detected']
        dominant_emotion = max(emotions_detected.items(), key=lambda x: x[1])[0] if any(emotions_detected.values()) else "unknown"
        employee_id = user_sessions[sid]['employee_id']
        if not employee_id:
            logger.warning(f"No employee_id for sid {sid}, skipping session save")
            return
        cursor.execute(
            """
            INSERT INTO sessions 
            (employee_id, duration_seconds, dominant_emotion, session_date)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (employee_id, session_duration, dominant_emotion, datetime.now().date())
        )
        session_id = cursor.fetchone()[0]
        user_sessions[sid]['session_id'] = session_id
        total_frames = active_cameras[sid]['frame_count']
        for emotion, count in emotions_detected.items():
            if count > 0:
                percentage = (count / total_frames) * 100
                cursor.execute(
                    """
                    INSERT INTO emotion_details 
                    (session_id, emotion, count, percentage)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, emotion, count, percentage)
                )
        conn.commit()
        logger.info(f"Saved session data for sid {sid}, session_id {session_id}, employee_id {employee_id}")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to save session data for sid {sid}: {e}")

@sio.event
async def start_camera(sid):
    logger.info(f"Starting camera for session: {sid}")
    if sid not in active_cameras or sid not in user_sessions:
        await sio.emit('error', {'message': 'Session not found'}, to=sid)
        return
    if not user_sessions[sid]['employee_id']:
        await sio.emit('error', {'message': 'Please log in before starting camera'}, to=sid)
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await sio.emit('error', {'message': 'Failed to open camera'}, to=sid)
        return
    active_cameras[sid]['cap'] = cap
    active_cameras[sid]['active'] = True
    active_cameras[sid]['start_time'] = time.time()
    active_cameras[sid]['frame_count'] = 0
    active_cameras[sid]['emotions_detected'] = {emotion: 0 for emotion in EMOTIONS}
    active_cameras[sid]['emotion_history'] = []
    history_size = 7
    dominant_emotion = "unknown"
    try:
        while active_cameras[sid]['active']:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame for session: {sid}")
                await sio.emit('error', {'message': 'Failed to read camera frame'}, to=sid)
                continue
            faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(30, 30))
            if len(faces) == 0:
                logger.debug(f"No faces detected in frame for session: {sid}")
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    img_tensor = transform(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prediction = ensemble.predict(img_tensor)
                        probs = prediction[0].cpu().numpy()
                        emotion_idx = np.argmax(probs)
                        confidence = probs[emotion_idx]
                        predicted_emotion = EMOTIONS[emotion_idx]
                    if confidence >= 0.4:
                        active_cameras[sid]['emotion_history'].append((predicted_emotion, confidence))
                        if len(active_cameras[sid]['emotion_history']) > history_size:
                            active_cameras[sid]['emotion_history'].pop(0)
                        counts = {}
                        for emotion, conf in active_cameras[sid]['emotion_history']:
                            counts[emotion] = counts.get(emotion, 0) + 1
                        dominant_emotion = max(counts, key=counts.get) if counts else "unknown"
                        if dominant_emotion and dominant_emotion != "unknown":
                            active_cameras[sid]['emotions_detected'][dominant_emotion] += 1
                            color = {
                                'angry': (0, 0, 255), 'disgust': (0, 140, 255), 'fear': (0, 69, 255),
                                'happy': (0, 255, 0), 'neutral': (255, 255, 0), 'sad': (255, 0, 0),
                                'surprise': (255, 0, 255)
                            }.get(dominant_emotion, (255, 255, 255))
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, f"{dominant_emotion}: {confidence:.2f}", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    logger.error(f"Error processing face for session {sid}: {e}")
                    continue
            active_cameras[sid]['frame_count'] += 1
            if active_cameras[sid]['frame_count'] % 100 == 0:
                logger.info(f"Processed {active_cameras[sid]['frame_count']} frames for session: {sid}")
                await save_session_data(sid)
            session_duration = time.time() - active_cameras[sid]['start_time']
            hours, rem = divmod(int(session_duration), 3600)
            mins, secs = divmod(rem, 60)
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                await sio.emit('frame', {
                    'frame': frame_base64,
                    'stats': active_cameras[sid]['emotions_detected'],
                    'status': dominant_emotion.capitalize() if dominant_emotion != "unknown" else 'No emotion detected',
                    'sessionTime': f"{hours:02}:{mins:02}:{secs:02}"
                }, to=sid)
            except Exception as e:
                logger.error(f"Error emitting frame for session {sid}: {e}")
                continue
            await asyncio.sleep(0.033)
    except Exception as e:
        logger.error(f"Error in camera loop for session {sid}: {e}")
        await sio.emit('error', {'message': f'Camera error: {str(e)}'}, to=sid)
    finally:
        cap.release()
        active_cameras[sid]['cap'] = None
        active_cameras[sid]['active'] = False
        await save_session_data(sid)
        logger.info(f"Camera stopped for session: {sid}, processed {active_cameras[sid]['frame_count']} frames")

@sio.event
async def stop_camera(sid):
    logger.info(f"Stopping camera for session: {sid}")
    if sid in active_cameras and active_cameras[sid]['active']:
        active_cameras[sid]['active'] = False
        if active_cameras[sid]['cap']:
            active_cameras[sid]['cap'].release()
            active_cameras[sid]['cap'] = None
        await save_session_data(sid)
        await sio.emit('camera_stopped', {'message': 'Camera stopped successfully'}, to=sid)

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    cap = cv2.VideoCapture(io.BytesIO(nparr))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, 100)
    emotions_detected = {emotion: 0 for emotion in EMOTIONS}
    emotion_history = []
    history_size = 7
    frame_count = 0
    start_time = time.time()
    while cap.isOpened() and frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(30, 30))
        for face_roi in [frame[y:y+h, x:x+w] for (x, y, w, h) in faces]:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = ensemble.predict(img_tensor)
                    probs = prediction[0].cpu().numpy()
                    emotion_idx = np.argmax(probs)
                    confidence = probs[emotion_idx]
                    predicted_emotion = EMOTIONS[emotion_idx]
                if confidence >= 0.4:
                    emotion_history.append((predicted_emotion, confidence))
                    if len(emotion_history) > history_size:
                        emotion_history.pop(0)
                    counts = {}
                    for emotion, conf in emotion_history:
                        counts[emotion] = counts.get(emotion, 0) + 1
                    dominant_emotion = max(counts, key=counts.get) if counts else None
                    if dominant_emotion:
                        emotions_detected[dominant_emotion] += 1
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                continue
        frame_count += 1
        if frames_to_process < total_frames:
            skip_frames = max(1, int(total_frames / frames_to_process) - 1)
            for _ in range(skip_frames):
                cap.read()
    cap.release()
    session_duration = time.time() - start_time
    dominant_emotion = max(emotions_detected.items(), key=lambda x: x[1])[0] if any(emotions_detected.values()) else "unknown"
    try:
        cursor.execute(
            """
            INSERT INTO sessions 
            (employee_id, duration_seconds, dominant_emotion, session_date)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            ("unknown", session_duration, dominant_emotion, datetime.now().date())
        )
        session_id = cursor.fetchone()[0]
        for emotion, count in emotions_detected.items():
            if count > 0:
                percentage = (count / frame_count) * 100
                cursor.execute(
                    """
                    INSERT INTO emotion_details 
                    (session_id, emotion, count, percentage)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, emotion, count, percentage)
                )
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to save video session data: {e}")
        raise HTTPException(status_code=500, detail="Error saving video session data")
    return {"message": "Video analysis complete", "stats": emotions_detected}

@app.get("/test_samples")
async def test_samples():
    test_dir = "C:\\Employee Welbeing through Emotion Detection\\The Solution\\grok\\data\\test"
    samples = []
    emotions_detected = {emotion: 0 for emotion in EMOTIONS}
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png')):
                    samples.append((os.path.join(class_dir, img_name), class_name))
    selected_samples = random.sample(samples, min(10, len(samples)))
    sample_images = []
    frame_count = len(selected_samples)
    start_time = time.time()
    for img_path, _ in selected_samples:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = ensemble.predict(img_tensor)
                probs = prediction[0].cpu().numpy()
                emotion_idx = np.argmax(probs)
                confidence = probs[emotion_idx]
                predicted_emotion = EMOTIONS[emotion_idx]
            if confidence >= 0.4:
                emotions_detected[predicted_emotion] += 1
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            sample_images.append({
                "data": base64.b64encode(img_byte_arr.getvalue()).decode('utf-8'),
                "label": predicted_emotion
            })
        except Exception as e:
            logger.error(f"Error processing test sample {img_path}: {e}")
            continue
    session_duration = time.time() - start_time
    dominant_emotion = max(emotions_detected.items(), key=lambda x: x[1])[0] if any(emotions_detected.values()) else "unknown"
    try:
        cursor.execute(
            """
            INSERT INTO sessions 
            (employee_id, duration_seconds, dominant_emotion, session_date)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            ("unknown", session_duration, dominant_emotion, datetime.now().date())
        )
        session_id = cursor.fetchone()[0]
        for emotion, count in emotions_detected.items():
            if count > 0:
                percentage = (count / frame_count) * 100
                cursor.execute(
                    """
                    INSERT INTO emotion_details 
                    (session_id, emotion, count, percentage)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, emotion, count, percentage)
                )
        conn.commit()
        logger.info(f"Saved test samples session, session_id {session_id}")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to save test samples session data: {e}")
    return {"message": "Test completed", "stats": emotions_detected, "sample_images": sample_images}

@app.get("/admin_stats")
async def admin_stats(department: str = "All", date_range: str = "Last 7 Days"):
    today = datetime.now().date()
    start_date = (
        (today - timedelta(days=7)).strftime("%Y-%m-%d") if date_range == "Last 7 Days" else
        (today - timedelta(days=30)).strftime("%Y-%m-%d") if date_range == "Last 30 Days" else
        "2000-01-01"
    )
    dept_condition = "" if department == "All" else f"AND e.department = '{department}'"
    try:
        conn.rollback()
        cursor.execute(f"""
            SELECT ed.emotion, SUM(ed.count) AS total_count
            FROM emotion_details ed
            JOIN sessions s ON ed.session_id = s.id
            JOIN employees e ON s.employee_id = e.employee_id
            WHERE s.session_date >= %s
            {dept_condition}
            GROUP BY ed.emotion
            ORDER BY total_count DESC
        """, (start_date,))
        emotion_stats = [{"status": row[0], "total": row[1]} for row in cursor.fetchall()]
        cursor.execute(f"""
            SELECT e.employee_id, e.department, COUNT(s.id) AS session_count,
                   AVG(s.duration_seconds) AS avg_duration,
                   MODE() WITHIN GROUP (ORDER BY s.dominant_emotion) AS dominant_emotion
            FROM employees e
            LEFT JOIN sessions s ON e.employee_id = e.employee_id
            WHERE s.session_date >= %s
            {dept_condition}
            GROUP BY e.employee_id, e.department
            ORDER BY session_count DESC
        """, (start_date,))
        employees = [
            {
                "employee_id": row[0],
                "department": row[1],
                "session_count": row[2],
                "avg_duration": row[3] / 60 if row[3] else 0,
                "dominant_emotion": row[4]
            } for row in cursor.fetchall()
        ]
        conn.commit()
        return {"compliance": emotion_stats, "employees": employees}
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Admin stats database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Admin stats error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving admin stats")