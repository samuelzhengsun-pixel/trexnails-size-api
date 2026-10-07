from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import math

# 兼容性导入 MediaPipe Hands 模块
import mediapipe as mp
try:
    mp_hands = mp.solutions.hands
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands

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

@app.get("/")
def home():
    return {"status": "TrexNails Hybrid Rotation Sizer Active"}

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
        if w > 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 1. 霍夫圆提取硬币位置与像素半径
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=40, 
            param1=40, 
            param2=18, 
            minRadius=int(w * 0.03), 
            maxRadius=int(w * 0.20)
        )

        coin_data = {"x": int(w * 0.2), "y": int(h * 0.5), "r": int(w * 0.08)}
        if circles is not None:
            circles = np.uint16(np.around(circles))
            best = circles[0][0]
            coin_data = {"x": int(best[0]), "y": int(best[1]), "r": int(best[2])}

        # 2. MediaPipe 检测手部节点并计算倾斜角度
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            model_complexity=1, 
            min_detection_confidence=0.1
        ) as hands:
            results = hands.process(img_rgb)
            
            nails_data = []
            if results.multi_hand_landmarks:
                pts = results.multi_hand_landmarks[0].landmark
                finger_pairs = [
                    {"name": "Index", "tip": 8, "dip": 7},
                    {"name": "Majeur", "tip": 12, "dip": 11},
                    {"name": "Annulaire", "tip": 16, "dip": 15},
                    {"name": "Auriculaire", "tip": 20, "dip": 19}
                ]
                
                for fp in finger_pairs:
                    tx, ty = pts[fp["tip"]].x * w, pts[fp["tip"]].y * h
                    dx, dy = pts[fp["dip"]].x * w, pts[fp["dip"]].y * h
                    
                    angle_rad = math.atan2(ty - dy, tx - dx)
                    angle_deg = math.degrees(angle_rad)
                    
                    len_px = math.sqrt((tx - dx)**2 + (ty - dy)**2)
                    cx = tx - (tx - dx) * 0.3
                    cy = ty - (ty - dy) * 0.3
                    
                    nails_data.append({
                        "name": fp["name"],
                        "cx": int(cx),
                        "cy": int(cy),
                        "angle": round(angle_deg, 1),
                        "width_px": int(len_px * 0.55)
                    })
            else:
                default_x = [int(w*0.4), int(w*0.5), int(w*0.6), int(w*0.7)]
                for i, name in enumerate(["Index", "Majeur", "Annulaire", "Auriculaire"]):
                    nails_data.append({
                        "name": name,
                        "cx": default_x[i],
                        "cy": int(h * 0.4),
                        "angle": -90.0,
                        "width_px": int(w * 0.05)
                    })

        return {
            "success": True,
            "img_width": w,
            "img_height": h,
            "coin_mm": real_coin_mm,
            "coin": coin_data,
            "nails": nails_data
        }
    except Exception as e:
        return {"success": False, "message": "Erreur d'analyse photo."}
