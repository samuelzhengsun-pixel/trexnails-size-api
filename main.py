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
    return {"status": "TrexNails Calibrated Nail Sizer Active"}

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

        # 保持高质量像素点数，最大宽度限制为 1000px
        h, w, _ = img.shape
        if w > 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 1. 霍夫圆算法提取硬币绝对物理标尺
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
        coin_px_diameter = best_coin[2] * 2

        # 🎯 校准 1：绝对物理标尺 (mm / px)
        mm_per_px = real_coin_mm / coin_px_diameter

        # 2. 手部骨骼定位指尖粗坐标
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Assurez-vous que vos 4 doigts sont bien à plat."
            }

        landmarks = results.multi_hand_landmarks[0].landmark

        # 🎯 校准 2：一维切线亮度梯度提取指甲边缘像素
        def get_calibrated_nail_px(tip_idx, dip_idx):
            p_tip = np.array([landmarks[tip_idx].x * w, landmarks[tip_idx].y * h])
            p_dip = np.array([landmarks[dip_idx].x * w, landmarks[dip_idx].y * h])
            
            vec = p_tip - p_dip
            vec_len = np.linalg.norm(vec)
            if vec_len == 0:
                return 13.0 / mm_per_px

            # 法线方向单位向量
            normal_vec = np.array([-vec[1], vec[0]]) / vec_len
            sample_center = p_dip + vec * 0.35
            
            scan_half_len = int(vec_len * 0.45)
            line_pts = []
            for i in range(-scan_half_len, scan_half_len):
                pt = sample_center + normal_vec * i
                px_x = int(np.clip(pt[0], 0, w - 1))
                px_y = int(np.clip(pt[1], 0, h - 1))
                line_pts.append(gray[px_y, px_x])

            if len(line_pts) > 5:
                grad = np.abs(np.diff(line_pts))
                threshold = np.max(grad) * 0.35
                peaks = np.where(grad > threshold)[0]
                if len(peaks) >= 2:
                    nail_px = peaks[-1] - peaks[0]
                    return nail_px

            return vec_len * 0.60

        # 🎯 校准 3：C-Curve 物理弧面增益 (1.04)
        c_curve = 1.04

        index_px = get_calibrated_nail_px(8, 7)
        middle_px = get_calibrated_nail_px(12, 11)
        ring_px = get_calibrated_nail_px(16, 15)
        pinky_px = get_calibrated_nail_px(20, 19)

        index_mm = index_px * mm_per_px * c_curve
        middle_mm = middle_px * mm_per_px * c_curve
        ring_mm = ring_px * mm_per_px * c_curve
        pinky_mm = pinky_px * mm_per_px * c_curve

        # 🎯 校准 4：对比官方尺码表合理边界保底（对应 XS-L 范围）
        index_mm = max(10.0, min(14.0, index_mm))
        middle_mm = max(11.0, min(15.0, middle_mm))
        ring_mm = max(10.0, min(14.0, ring_mm))
        pinky_mm = max(8.0, min(12.0, pinky_mm))

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
