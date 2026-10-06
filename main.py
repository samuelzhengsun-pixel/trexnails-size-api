from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COIN_SIZES_MM = {
    "2e": 25.75,
    "1e": 23.25,
    "50ct": 24.25
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.1
)

@app.get("/")
def home():
    return {"status": "TrexNails AI Service Active"}

@app.post("/api/scan-nails")
async def scan_nails(
    file: UploadFile = File(...),
    coin_type: str = Form("2e")
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"success": False, "message": "Photo invalide."}

        h, w, _ = img.shape
        if w > 800:
            scale = 800.0 / w
            img = cv2.resize(img, (800, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 1. 精确硬币直径提取 (像素标尺)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30, 
            param1=50, 
            param2=18, 
            minRadius=10, 
            maxRadius=300
        )

        if circles is None:
            return {
                "success": False, 
                "message": f"Pièce ({coin_type.upper()}) non détectée ! Posez la pièce bien à plat sur la table."
            }

        circles = np.uint16(np.around(circles))
        best_coin = circles[0][0]
        coin_px_diameter = best_coin[2] * 2

        # 物理标尺 (mm / px)
        mm_per_px = real_coin_mm / coin_px_diameter

        # 2. AI 手部骨骼横向极值定位
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Assurez-vous que vos 4 doigts sont bien à plat."
            }

        landmarks = results.multi_hand_landmarks[0].landmark

        # 计算食指指节 (Landmark 5/8) 到 小指指节 (Landmark 17/20) 的横向跨越总像素值
        index_x = landmarks[8].x * w
        index_y = landmarks[8].y * h
        pinky_x = landmarks[20].x * w
        pinky_y = landmarks[20].y * h

        hand_horizontal_span_px = math.sqrt((index_x - pinky_x)**2 + (index_y - pinky_y)**2)

        # C-Curve 微调系数 (1.02)
        c_curve = 1.02

        # 🎯 精确横向物理分割比例 (对应 14, 15, 14, 12 mm 实测模型)
        index_mm = (hand_horizontal_span_px * 0.270) * mm_per_px * c_curve
        middle_mm = (hand_horizontal_span_px * 0.290) * mm_per_px * c_curve
        ring_mm = (hand_horizontal_span_px * 0.270) * mm_per_px * c_curve
        pinky_mm = (hand_horizontal_span_px * 0.230) * mm_per_px * c_curve

        # 合理区间校准
        index_mm = max(8.5, min(16.5, index_mm))
        middle_mm = max(9.0, min(17.0, middle_mm))
        ring_mm = max(8.5, min(16.5, ring_mm))
        pinky_mm = max(6.5, min(13.5, pinky_mm))

        return {
            "success": True,
            "coin_used": coin_type.upper(),
            "measures": {
                "index": round(index_mm, 1),
                "middle": round(middle_mm, 1),
                "ring": round(ring_mm, 1),
                "pinky": round(pinky_mm, 1)
            }
        }
    except Exception as e:
        return {"success": False, "message": "Erreur d'analyse."}
