from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ultralytics import YOLO
import google.generativeai as genai
import shutil

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
model = YOLO('models/best.pt')  # Use custom trained model

# Load API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    raise ValueError("❌ ERROR: GEMINI_API_KEY is not set. Please configure it before running.")

def analyze_with_gemini(player_data):
    if len(player_data['positions']) < 2 or player_data['distance'] == 0:
        return "Not enough movement data to analyze."
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Summarize the player's performance in 4-5 points including attacking, defending, and ball control based on: {player_data}"
    
    for _ in range(3):  # Retry up to 3 times
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}, retrying...")
            time.sleep(2)
    
    return "Error: Gemini API unavailable."

# Placeholder for real-world scaling (adjust based on known field dimensions)
pixels_per_meter = 50  # Approximate conversion factor

# Tracker class for better player tracking
class Tracker:
    def __init__(self):
        self.tracked_players = {}
        self.fps = 30  # Update based on actual video FPS
    
    def update(self, detections, player_id):
        for detection in detections:
            x1, y1, x2, y2 = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if player_id not in self.tracked_players:
                self.tracked_players[player_id] = {
                    'positions': [(center_x, center_y)],
                    'distance': 0.0,
                    'attack_time': 0,
                    'defense_time': 0
                }
            else:
                prev_x, prev_y = self.tracked_players[player_id]['positions'][-1]
                dx = center_x - prev_x
                dy = center_y - prev_y
                distance = np.sqrt(dx**2 + dy**2)
                self.tracked_players[player_id]['distance'] += distance
                self.tracked_players[player_id]['positions'].append((center_x, center_y))

                # Define midfield line (assumed center of image)
                midfield_line = 640 / 2  # Adjust based on actual field dimensions
                if center_x > midfield_line:
                    self.tracked_players[player_id]['attack_time'] += 1
                else:
                    self.tracked_players[player_id]['defense_time'] += 1
        return self.tracked_players

# Function to calculate player speed
def calculate_speed(player_data, fps=30, pixels_per_meter=50):
    total_distance_pixels = player_data['distance']
    total_time_seconds = len(player_data['positions']) / fps
    if total_time_seconds == 0:
        return 0
    speed_mps = (total_distance_pixels / pixels_per_meter) / total_time_seconds
    speed_kmh = speed_mps * 3.6
    return speed_kmh

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    video_path = f"videos/{file.filename}"
    os.makedirs("videos", exist_ok=True)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Video uploaded successfully", "video_path": video_path}

@app.post("/track/")
async def track_player(video_path: str = Form(...), player_id: int = Form(...)):
    cap = cv2.VideoCapture(video_path)
    tracker = Tracker()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, classes=0, conf=0.3)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        
        tracked_players = tracker.update(boxes, player_id)
    
    cap.release()
    
    if player_id in tracker.tracked_players:
        data = tracker.tracked_players[player_id]
        speed_kmh = calculate_speed(data, tracker.fps, pixels_per_meter)
        distance_m = data['distance'] / pixels_per_meter
        analysis = analyze_with_gemini(data)
        
        total_time = len(data['positions'])
        attack = (data['attack_time'] / total_time) * 100 if total_time > 0 else 0
        defense = (data['defense_time'] / total_time) * 100 if total_time > 0 else 0
        ball_control = (100 - abs(attack - defense))
        
        return {
            "player_id": player_id,
            "speed_kmh": speed_kmh,
            "distance_m": distance_m,
            "attack": attack,
            "defense": defense,
            "ball_control": ball_control,
            "analysis": analysis
        }
    
    return {"error": "Player not found in video"}

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    video_path = f"videos/{file.filename}"
    os.makedirs("videos", exist_ok=True)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"status": "Processed successfully", "video_path": video_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
