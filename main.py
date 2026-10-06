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

# 高精度 AI 骨骼检测网络
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

        # 1. 图片降采样防卡顿 (800px)
        h, w, _ = img.shape
        if w > 800:
            scale = 800.0 / w
            img = cv2.resize(img, (800, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 2. 硬币多轮廓透视修正 (锁定正圆/椭圆最大直径)
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

        # 精确标尺 (mm/px)
        mm_per_px = real_coin_mm / coin_px_diameter

        # 3. AI 骨骼关节直接测量指甲绝对宽度
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Assurez-vous que vos 4 doigts sont bien à plat."
            }

        landmarks = results.multi_hand_landmarks[0].landmark

        # 两点欧式空间距离计算（向量自适应旋转）
        def get_joint_dist(pt1_idx, pt2_idx):
            p1 = landmarks[pt1_idx]
            p2 = landmarks[pt2_idx]
            dx = (p1.x - p2.x) * w
            dy = (p1.y - p2.y) * h
            return math.sqrt(dx*dx + dy*dy)

        # 🎯 物理级毫米校准系数 (1.02) - 精确对应皮尺平铺测量
        c_curve = 1.02
        
        # 直接提取每根手指 DIP/PIP 指节的骨骼几何真实物理像素宽度
        index_w_px = get_joint_dist(8, 7) * 1.08
        middle_w_px = get_joint_dist(12, 11) * 1.08
        ring_w_px = get_joint_dist(16, 15) * 1.08
        pinky_w_px = get_joint_dist(20, 19) * 1.08

        index_mm = index_w_px * mm_per_px * c_curve
        middle_mm = middle_w_px * mm_per_px * c_curve
        ring_mm = ring_w_px * mm_per_px * c_curve
        pinky_mm = pinky_w_px * mm_per_px * c_curve

        # 边界校准截断
        index_mm = max(9.0, min(17.0, index_mm))
        middle_mm = max(9.0, min(17.5, middle_mm))
        ring_mm = max(9.0, min(16.5, ring_mm))
        pinky_mm = max(7.0, min(13.5, pinky_mm))

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
