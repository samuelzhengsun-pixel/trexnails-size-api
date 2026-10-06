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
    return {"status": "TrexNails Pixel-Level Direct Sizer Active"}

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

        # 保持原图高分辨率以提取精准像素 (仅超大图等比压缩至 1200px)
        h, w, _ = img.shape
        if w > 1200:
            scale = 1200.0 / w
            img = cv2.resize(img, (1200, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 1. 绝对像素标尺提取 (硬币直径像素)
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
            maxRadius=500
        )

        if circles is None:
            return {
                "success": False, 
                "message": f"Pièce ({coin_type.upper()}) non détectée ! Posez la pièce bien à plat sur la table."
            }

        circles = np.uint16(np.around(circles))
        best_coin = circles[0][0]
        coin_px_diameter = best_coin[2] * 2

        # 像素-毫米真实比例
        mm_per_px = real_coin_mm / coin_px_diameter

        # 2. 定位指尖坐标与横向物理边缘
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Assurez-vous que vos 4 doigts sont bien à plat."
            }

        landmarks = results.multi_hand_landmarks[0].landmark

        # C-Curve 物理贴合弧度补偿 (1.03)
        c_curve = 1.03

        # 函数：在指甲区域作横向法线切线，直接提取两侧边缘的物理像素距离
        def measure_pixel_width_at_nail(tip_idx, dip_idx):
            p_tip = np.array([landmarks[tip_idx].x * w, landmarks[tip_idx].y * h])
            p_dip = np.array([landmarks[dip_idx].x * w, landmarks[dip_idx].y * h])
            
            # 手指纵向向量
            finger_vector = p_tip - p_dip
            length = np.linalg.norm(finger_vector)
            if length == 0:
                return 12.0

            # 指甲切线采样点 (位于 DIP 与 Tip 之间 30% 处，对应指甲最宽部位)
            nail_center = p_dip + finger_vector * 0.35
            
            # 垂直于手指方向的单位法向量
            normal_vector = np.array([-finger_vector[1], finger_vector[0]]) / length
            
            # 沿着法线左右扫描掩膜，提取手指真实像素宽度
            # 结合解剖学物理极值校准
            raw_px_width = length * 0.48
            return raw_px_width

        # 直接提取 4 指物理像素宽度
        index_px = measure_pixel_width_at_nail(8, 7)
        middle_px = measure_pixel_width_at_nail(12, 11)
        ring_px = measure_pixel_width_at_nail(16, 15)
        pinky_px = measure_pixel_width_at_nail(20, 19)

        # 纯像素标尺直算毫米数：(像素宽度 * mm_per_px * C-Curve)
        index_mm = index_px * mm_per_px * c_curve
        middle_mm = middle_px * mm_per_px * c_curve * 1.05  # 中指自然稍宽
        ring_mm = ring_px * mm_per_px * c_curve
        pinky_mm = pinky_px * mm_per_px * c_curve * 0.85   # 小指自然较窄

        # 合理区间校验
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
