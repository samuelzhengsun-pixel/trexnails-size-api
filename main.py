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
    min_detection_confidence=0.3
)

@app.get("/")
def home():
    return {"status": "TrexNails Strict Commercial Sizer Active"}

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
            return {"success": False, "message": "Photo invalide ou floue. Veuillez reprendre une photo claire."}

        h, w, _ = img.shape
        if w > 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 1. 硬币精确度严肃拦截
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
                "message": f"Impossible de détecter la pièce de {coin_type.upper()}. Veuillez poser la pièce bien à plat à côté de vos doigts sur une surface dégagée."
            }

        circles = np.uint16(np.around(circles))
        best_coin = circles[0][0]
        coin_px_diameter = best_coin[2] * 2

        if coin_px_diameter < 20:
            return {
                "success": False,
                "message": "La pièce est trop petite ou trop éloignée. Rapprochez votre appareil photo."
            }

        mm_per_px = real_coin_mm / coin_px_diameter

        # 2. 手部与指甲姿态严肃校验
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False, 
                "message": "Main non détectée. Veillez à bien poser vos 4 doigts à plat sous un bon éclairage."
            }

        landmarks = results.multi_hand_landmarks[0].landmark

        # 3. 真实物理切线采样（若边缘对比度不足直接返回 None，绝不瞎猜）
        def get_strict_nail_px(tip_idx, dip_idx):
            p_tip = np.array([landmarks[tip_idx].x * w, landmarks[tip_idx].y * h])
            p_dip = np.array([landmarks[dip_idx].x * w, landmarks[dip_idx].y * h])
            
            vec = p_tip - p_dip
            vec_len = np.linalg.norm(vec)
            if vec_len == 0:
                return None

            normal_vec = np.array([-vec[1], vec[0]]) / vec_len
            sample_center = p_dip + vec * 0.35
            
            scan_half_len = int(vec_len * 0.55)
            line_pts = []
            for i in range(-scan_half_len, scan_half_len):
                pt = sample_center + normal_vec * i
                px_x = int(np.clip(pt[0], 0, w - 1))
                px_y = int(np.clip(pt[1], 0, h - 1))
                line_pts.append(gray[px_y, px_x])

            if len(line_pts) > 5:
                grad = np.abs(np.diff(line_pts))
                threshold = np.max(grad) * 0.18
                peaks = np.where(grad > threshold)[0]
                
                # 严苛校验：指甲两侧必须都存在清晰的物理边缘峰值
                if len(peaks) >= 2:
                    nail_px = peaks[-1] - peaks[0]
                    # 检查像素合理性（避免抓到手指外边缘）
                    if 10 < nail_px < (vec_len * 1.3):
                        return nail_px

            # 边缘不清晰或没找到：严肃返回 None，拒绝猜数据
            return None

        # 物理弧面展开系数 (3D 真实弧长标准)
        c_curve_gain = 1.05

        index_px = get_strict_nail_px(8, 7)
        middle_px = get_strict_nail_px(12, 11)
        ring_px = get_strict_nail_px(16, 15)
        pinky_px = get_strict_nail_px(20, 19)

        # 只要有任何一根手指无法清晰定位指甲边缘，绝对不乱给，直接阻断报错提示顾客！
        if index_px is None or middle_px is None or ring_px is None or pinky_px is None:
            return {
                "success": False,
                "message": "Contours des ongles flous. Veuillez reprendre la photo sous un éclairage direct, sur ongles nus et sans ombre."
            }

        # 纯真实物理转换（无硬编码截断，全由边缘切线与硬币标尺直接算得）
        index_mm = index_px * mm_per_px * c_curve_gain
        middle_mm = middle_px * mm_per_px * c_curve_gain
        ring_mm = ring_px * mm_per_px * c_curve_gain
        pinky_mm = pinky_px * mm_per_px * c_curve_gain

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
        return {"success": False, "message": "Erreur d'analyse. Merci de re-prendre une photo bien nette."}
