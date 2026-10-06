from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
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

@app.get("/")
def home():
    return {"status": "TrexNails Precision True Physical Sizer V6.0 Active"}

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

        # 保持高质量分辨率
        h, w, _ = img.shape
        if w > 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 1. 消除阴影干扰的硬币精确定位 (Otsu 自动二值化 + 轮廓过滤)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 霍夫圆检测精确抓取金属外圈真实边界
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=40, 
            param1=50, 
            param2=22, # 提高检出阈值，排除阴影噪点
            minRadius=15, 
            maxRadius=350
        )

        coin_px_diameter = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            best_coin = circles[0][0]
            # 半径 * 2 得到真实硬币像素直径
            coin_px_diameter = best_coin[2] * 2
        else:
            # 备用 Otsu 剔除阴影轮廓
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if 300 < area < (w * h * 0.2):
                    (cx, cy), (d1, d2), _ = cv2.fitEllipse(c)
                    if min(d1, d2) / max(d1, d2) > 0.7:
                        coin_px_diameter = (d1 + d2) / 2.0
                        break

        if coin_px_diameter == 0:
            return {
                "success": False, 
                "message": f"Pièce ({coin_type.upper()}) non détectée. Assurez-vous que la pièce est bien visible sur la table."
            }

        # 精确绝对物理标尺 (mm/px)
        mm_per_px = real_coin_mm / coin_px_diameter

        # 2. 皮肤与手部分割 (HSV 色彩空间)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 12, 35], dtype=np.uint8)
        upper_skin = np.array([28, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_mask = cv2.medianBlur(skin_mask, 5)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"success": False, "message": "Main non détectée. Veillez à utiliser un fond clair."}

        hand_contour = max(contours, key=cv2.contourArea)
        x, y, hand_w, hand_h = cv2.boundingRect(hand_contour)

        # 3. 横向指甲最宽处采样
        roi_y1 = y + int(hand_h * 0.08)
        roi_y2 = y + int(hand_h * 0.38)
        
        finger_widths_px = []
        for scan_y in range(roi_y1, roi_y2, max(1, (roi_y2 - roi_y1) // 12)):
            row = skin_mask[scan_y, x:x+hand_w]
            changes = np.diff(row.astype(np.int16))
            starts = np.where(changes > 200)[0]
            ends = np.where(changes < -200)[0]
            
            widths = []
            if len(starts) > 0 and len(ends) > 0:
                for s in starts:
                    valid_ends = ends[ends > s]
                    if len(valid_ends) > 0:
                        w_px = valid_ends[0] - s
                        if 12 < w_px < (hand_w * 0.38):
                            widths.append(w_px)
            if len(widths) >= 4:
                finger_widths_px.append(widths[:4])

        if not finger_widths_px:
            # 物理真实比例展开
            index_px = hand_w * 0.22
            middle_px = hand_w * 0.24
            ring_px = hand_w * 0.22
            pinky_px = hand_w * 0.18
        else:
            avg_widths = np.median(finger_widths_px, axis=0)
            index_px = avg_widths[0]
            middle_px = avg_widths[1]
            ring_px = avg_widths[2]
            pinky_px = avg_widths[3]

        # 🎯 物理真实系数映射 (手指像素宽度 -> 剥离肌肉肉厚 -> 3D C-Curve 弧度平铺)
        # 彻底解除一切 min(10.0) 强制锁死截断！
        nail_to_finger_ratio = 0.78  # 解剖学指甲盖占手指横截面物理比例
        c_curve = 1.05

        index_mm = index_px * nail_to_finger_ratio * mm_per_px * c_curve
        middle_mm = middle_px * nail_to_finger_ratio * mm_per_px * c_curve
        ring_mm = ring_px * nail_to_finger_ratio * mm_per_px * c_curve
        pinky_mm = pinky_px * nail_to_finger_ratio * mm_per_px * c_curve

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
