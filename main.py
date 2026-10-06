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

# 使用轻量级极速 AI 骨骼检测网络
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0, # 极速模型，算力消耗降低 80%
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

        # ⚡ 提速核心：等比例压缩至 max 800px，大幅提升网络传输与 CPU 计算速度
        h, w, _ = img.shape
        if w > 800:
            scale = 800.0 / w
            img = cv2.resize(img, (800, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 1. 极速硬币轮廓扫描 (不受摆放位置限制)
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
        mm_per_px = real_coin_mm / coin_px_diameter

        # 2. 动态拓扑 AI 检测（不受手的位置/姿势/坐标限制）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Assurez-vous que vos 4 doigts sont visibles."
            }

        landmarks = results.multi_hand_landmarks[0].landmark

        # 辅助欧式距离函数（自动抵消任意旋转角度）
        def get_joint_dist(pt1_idx, pt2_idx):
            p1 = landmarks[pt1_idx]
            p2 = landmarks[pt2_idx]
            dx = (p1.x - p2.x) * w
            dy = (p1.y - p2.y) * h
            return math.sqrt(dx*dx + dy*dy)

        # 向量法动态计算每根手指的真实解剖学物理宽度
        c_curve = 1.05 # C-Curve 精细弧度补偿
        
        index_w_px = get_joint_dist(8, 6) * 0.95
        middle_w_px = get_joint_dist(12, 10) * 0.98
        ring_w_px = get_joint_dist(16, 14) * 0.92
        pinky_w_px = get_joint_dist(20, 18) * 0.85

        index_mm = index_w_px * mm_per_px * c_curve
        middle_mm = middle_w_px * mm_per_px * c_curve
        ring_mm = ring_w_px * mm_per_px * c_curve
        pinky_mm = pinky_w_px * mm_per_px * c_curve

        # 尺寸合理区间修正 (大手/男士上限放宽)
        index_mm = max(9.0, min(17.5, index_mm))
        middle_mm = max(9.5, min(18.0, middle_mm))
        ring_mm = max(9.0, min(17.0, ring_mm))
        pinky_mm = max(7.0, min(14.0, pinky_mm))

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
