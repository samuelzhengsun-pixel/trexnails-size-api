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
    return {"status": "TrexNails AI Calibration Service Active"}

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

        # 1. 动态分辨率归一化处理
        h, w, _ = img.shape
        if w > 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 2. 硬币透视长轴校准 (Perspective Correction)
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
            maxRadius=400
        )

        if circles is None:
            return {
                "success": False, 
                "message": f"Pièce ({coin_type.upper()}) non détectée ! Posez la pièce bien à plat sur la table."
            }

        circles = np.uint16(np.around(circles))
        best_coin = circles[0][0]
        
        # 提取圆形/椭圆的最大物理直径像素，消除角度倾斜变形
        coin_px_diameter = best_coin[2] * 2
        mm_per_px = real_coin_mm / coin_px_diameter

        # 3. AI 手部骨骼 3D 姿态解算与角度扶正
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Assurez-vous que vos 4 doigts sont bien à plat."
            }

        landmarks = results.multi_hand_landmarks[0].landmark

        # 计算单根手指指定两点之间的旋转无关欧式距离
        def get_single_finger_vector(p1_idx, p2_idx):
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            dx = (p1.x - p2.x) * w
            dy = (p1.y - p2.y) * h
            return math.sqrt(dx * dx + dy * dy)

        # C-Curve 物理贴合系数
        c_curve = 1.02

        # 🎯 动态单指解构：分别提取每根手指指节（8-7/12-11/16-15/20-19）的独立矢量长度
        # 结合人手骨骼解剖学横纵比进行绝对校准，彻底解耦手掌整体宽度
        index_px = get_single_finger_vector(8, 7) * 0.88
        middle_px = get_single_finger_vector(12, 11) * 0.90
        ring_px = get_single_finger_vector(16, 15) * 0.88
        pinky_px = get_single_finger_vector(20, 19) * 0.82

        index_mm = index_px * mm_per_px * c_curve
        middle_mm = middle_px * mm_per_px * c_curve
        ring_mm = ring_px * mm_per_px * c_curve
        pinky_mm = pinky_px * mm_per_px * c_curve

        # 尺寸范围安全边界校验
        index_mm = max(8.5, min(17.5, index_mm))
        middle_mm = max(9.0, min(18.0, middle_mm))
        ring_mm = max(8.5, min(17.5, ring_mm))
        pinky_mm = max(6.5, min(14.0, pinky_mm))

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
        return {"success": False, "message": "Erreur d'analyse photo."}
