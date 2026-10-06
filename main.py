from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_hands_solution = mp.solutions.hands
hands = mp_hands_solution.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.3, # 降低置信度阈值，防止硬币稍微挡住就报错
    min_tracking_confidence=0.3
)

@app.get("/")
def home():
    return {"status": "TrexNails AI Service Active"}

@app.post("/api/scan-nails")
async def scan_nails(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"success": False, "message": "Photo invalide ou floue."}

        # 1. 硬币轮廓检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=80, 
            param1=80, 
            param2=25, 
            minRadius=20, 
            maxRadius=350
        )

        if circles is None:
            return {
                "success": False, 
                "message": "Pièce non détectée ! Posez la pièce DE CÔTÉ sur la table (ne la posez pas sur vos doigts)."
            }

        circles = np.uint16(np.around(circles))
        coin_px_radius = circles[0][0][2]
        coin_px_diameter = coin_px_radius * 2
        mm_per_px = 25.75 / coin_px_diameter # 2 Euro 标尺

        # 2. AI 手部 4 指识别
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Posez vos 4 doigts bien à plat sans couvrir les articulations avec la pièce."
            }

        h, w, _ = img.shape
        landmarks = results.multi_hand_landmarks[0].landmark

        # C-Curve 弧度 1.07x
        c_curve = 1.07
        index_w_px = abs(landmarks[8].x - landmarks[6].x) * w
        middle_w_px = abs(landmarks[12].x - landmarks[10].x) * w
        ring_w_px = abs(landmarks[16].x - landmarks[14].x) * w
        pinky_w_px = abs(landmarks[20].x - landmarks[18].x) * w

        index_mm = max(9.0, min(15.0, index_w_px * mm_per_px * c_curve * 1.35))
        middle_mm = max(9.0, min(15.0, middle_w_px * mm_per_px * c_curve * 1.35))
        ring_mm = max(9.0, min(15.0, ring_w_px * mm_per_px * c_curve * 1.35))
        pinky_mm = max(7.0, min(13.0, pinky_w_px * mm_per_px * c_curve * 1.35))

        return {
            "success": True,
            "measures": {
                "index": round(index_mm, 1),
                "middle": round(middle_mm, 1),
                "ring": round(ring_mm, 1),
                "pinky": round(pinky_mm, 1)
            }
        }
    except Exception as e:
        return {"success": False, "message": "Erreur d'analyse. Merci d'utiliser une photo claire sur ongles nus."}
