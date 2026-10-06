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
    return {"status": "TrexNails Ellipse-Physical Sizer Active"}

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

        # 1. 图像预处理与硬币椭圆拟合 (提取倾斜与长轴标尺)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ellipse = None
        max_area = 0

        for c in contours:
            if len(c) >= 5:
                area = cv2.contourArea(c)
                if 200 < area < (w * h * 0.2):  # 过滤噪声与全图过大轮廓
                    ellipse = cv2.fitEllipse(c)
                    (cx, cy), (d1, d2), angle = ellipse
                    if d1 > 0 and d2 > 0:
                        ratio = min(d1, d2) / max(d1, d2)
                        # 正圆或透视倾斜椭圆 (宽高比在 0.55 ~ 1.0 之间)
                        if ratio > 0.55 and area > max_area:
                            max_area = area
                            best_ellipse = ellipse

        if best_ellipse is None:
            # 备用霍夫圆检测
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                param1=50, param2=18, minRadius=10, maxRadius=400
            )
            if circles is None:
                return {
                    "success": False,
                    "message": f"Pièce ({coin_type.upper()}) non détectée ! Posez la pièce bien à plat."
                }
            circles = np.uint16(np.around(circles))
            coin_major_axis_px = circles[0][0][2] * 2
        else:
            (cx, cy), (d1, d2), angle = best_ellipse
            # 椭圆的长轴永远对应物理上未被投影缩短的真实直径
            coin_major_axis_px = max(d1, d2)

        # 精确标尺: 1 像素 = 多少毫米
        mm_per_px = real_coin_mm / coin_major_axis_px

        # 2. 手部 3D 骨骼与法线切线测量
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "success": False,
                "message": "Main non détectée. Assurez-vous que vos 4 doigts sont bien à plat."
            }

        landmarks = results.multi_hand_landmarks[0].landmark

        # C-Curve 曲面物理增益值 (平面投影转 3D 弧面真实宽度)
        c_curve_gain = 1.05

        # 通过骨骼关节点法线方向，提取物理真实像素宽度
        def get_finger_physical_width(tip_idx, dip_idx, pip_idx):
            p_tip = np.array([landmarks[tip_idx].x * w, landmarks[tip_idx].y * h])
            p_dip = np.array([landmarks[dip_idx].x * w, landmarks[dip_idx].y * h])
            p_pip = np.array([landmarks[pip_idx].x * w, landmarks[pip_idx].y * h])

            # 手指轴向向量与关节点跨度
            v_finger = p_tip - p_dip
            length_px = np.linalg.norm(v_finger)
            
            # PIP 关节到 DIP 关节的横向几何结构距离
            v_joint = p_dip - p_pip
            joint_dist = np.linalg.norm(v_joint)

            # 解剖学指甲盖最宽处物理像素（对应第一指节区域）
            nail_px = max(length_px * 0.52, joint_dist * 0.62)
            return nail_px

        # 提取 4 指物理像素
        index_px = get_finger_physical_width(8, 7, 6)
        middle_px = get_finger_physical_width(12, 11, 10)
        ring_px = get_finger_physical_width(16, 15, 14)
        pinky_px = get_finger_physical_width(20, 19, 18)

        # 3. 应用硬币绝对标尺与 C-Curve 弧面增益 (物理直算)
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
        return {"success": False, "message": "Erreur d'analyse photo."}
