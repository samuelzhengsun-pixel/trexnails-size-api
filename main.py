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
        # 统一尺寸
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

        coin_data = {"x": int(w * 0.2), "y": int(h * 0.5), "r": int(w * 0.08)} # 默认初始化
        if circles is not None:
            circles = np.uint16(np.around(circles))
            best = circles[0][0]
            coin_data = {"x": int(best[0]), "y": int(best[1]), "r": int(best[2])}

        # 2. MediaPipe 检测 21 个 3D 节点并计算手指倾斜角度 (Arbitrary Angle Rotation)
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
                # 4 根手指节点对 (Tip, DIP)
                finger_pairs = [
                    {"name": "Index", "tip": 8, "dip": 7},
                    {"name": "Majeur", "tip": 12, "dip": 11},
                    {"name": "Annulaire", "tip": 16, "dip": 15},
                    {"name": "Auriculaire", "tip": 20, "dip": 19}
                ]
                
                for fp in finger_pairs:
                    tx, ty = pts[fp["tip"]].x * w, pts[fp["tip"]].y * h
                    dx, dy = pts[fp["dip"]].x * w, pts[fp["dip"]].y * h
                    
                    # 计算任意倾斜角度 angle (弧度与角度)
                    angle_rad = math.atan2(ty - dy, tx - dx)
                    angle_deg = math.degrees(angle_rad)
                    
                    # 估计指甲盖中心与宽度范围
                    len_px = math.sqrt((tx - dx)**2 + (ty - dy)**2)
                    cx = tx - (tx - dx) * 0.3
                    cy = ty - (ty - dy) * 0.3
                    
                    nails_data.append({
                        "name": fp["name"],
                        "cx": int(cx),
                        "cy": int(cy),
                        "angle": round(angle_deg, 1),
                        "width_px": int(len_px * 0.55) # AI 自动推荐的初始像素框宽度
                    })
            else:
                # 若没找到手，给出一组默认垂直微调框，绝不报错卡死
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
