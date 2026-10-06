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

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.6
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

        # 1. 灰度化与硬币检测 (霍夫圆与椭圆校正)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=100, 
            param1=100, 
            param2=30, 
            minRadius=25, 
            maxRadius=300
        )

        if circles is None:
            return {
                "success": False, 
                "message": "Pièce de monnaie non détectée. Posez une pièce de 2€/1€ à côté des doigts."
            }

        circles = np.uint16(np.around(circles))
        coin_px_radius = circles[0][0][2]
        coin_px_diameter = coin_px_radius * 2

        # 2€ 硬币 25.75mm 标尺
        mm_per_px = 25.75 / coin_px_diameter

        # 2. MediaPipe AI 识别手部关键点
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Veuillez poser vos 4 doigts bien à plat."
            }

        h, w, _ = img.shape
        landmarks = results.multi_hand_landmarks[0].landmark

        # 算各手指宽度（含 C-Curve 1.07 弧度校正）
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
        return {"success": False, "message": "Erreur d'analyse. Veillez à utiliser une photo claire sur ongles nus."}
