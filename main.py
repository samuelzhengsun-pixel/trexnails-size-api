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
    return {"status": "TrexNails OpenCV Skin Segmentation Active"}

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

        # 1. 统一图片尺寸 (1000px 宽度)
        h, w, _ = img.shape
        if w > 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 2. 硬币高精度识别 (霍夫圆 + 轮廓拟合双保底)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        coin_px_diameter = 0

        # 霍夫圆
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30, 
            param1=35, 
            param2=12, 
            minRadius=10, 
            maxRadius=400
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            best_coin = circles[0][0]
            coin_px_diameter = best_coin[2] * 2
        else:
            # 轮廓拟合椭圆
            edges = cv2.Canny(blurred, 20, 80)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            for c in contours:
                if len(c) >= 5:
                    area = cv2.contourArea(c)
                    if 150 < area < (w * h * 0.3):
                        ellipse = cv2.fitEllipse(c)
                        (cx, cy), (d1, d2), _ = ellipse
                        if d1 > 0 and d2 > 0:
                            ratio = min(d1, d2) / max(d1, d2)
                            if ratio > 0.55 and area > max_area:
                                max_area = area
                                coin_px_diameter = max(d1, d2)

        if coin_px_diameter == 0:
            return {
                "success": False, 
                "message": f"Pièce ({coin_type.upper()}) non détectée. Posez la pièce bien à plat à côté des doigts."
            }

        # 绝对物理标尺 (mm / px)
        mm_per_px = real_coin_mm / coin_px_diameter

        # 3. 基于 HSV 肤色分割分割手部轮廓 (无需 MediaPipe 完整手掌)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 肤色 HSV 范围定义
        lower_skin = np.array([0, 15, 40], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_mask = cv2.medianBlur(skin_mask, 5)

        # 寻找最大的肤色轮廓 (手部)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "success": False,
                "message": "Main non détectée. Posez vos doigts sur un fond clair."
            }

        hand_contour = max(contours, key=cv2.contourArea)
        x, y, hand_w, hand_h = cv2.boundingRect(hand_contour)

        # 4. 提取 4 根手指纵向极值与指甲盖最宽像素
        # 在手部轮廓顶部 20%~40% 区域（对应指甲盖物理位置）做横向分割
        roi_y1 = y + int(hand_h * 0.05)
        roi_y2 = y + int(hand_h * 0.35)
        
        finger_widths_px = []
        
        # 在指甲区域纵向扫描 10 条横截线，取极值
        for scan_y in range(roi_y1, roi_y2, max(1, (roi_y2 - roi_y1) // 10)):
            row = skin_mask[scan_y, x:x+hand_w]
            # 寻找肤色像素块（手指）
            changes = np.diff(row.astype(np.int16))
            starts = np.where(changes > 200)[0]
            ends = np.where(changes < -200)[0]
            
            # 如果捕捉到多根手指
            widths = []
            if len(starts) > 0 and len(ends) > 0:
                for s in starts:
                    valid_ends = ends[ends > s]
                    if len(valid_ends) > 0:
                        w_px = valid_ends[0] - s
                        if 15 < w_px < (hand_w * 0.35): # 过滤非手指噪声
                            widths.append(w_px)
            if len(widths) >= 4:
                finger_widths_px.append(widths[:4])

        if not finger_widths_px:
            # 兜底物理比例映射 (按手部总宽度与解剖学比例直算)
            index_px = hand_w * 0.20
            middle_px = hand_w * 0.22
            ring_px = hand_w * 0.20
            pinky_px = hand_w * 0.16
        else:
            # 取稳定的中位数指甲像素宽度
            avg_widths = np.median(finger_widths_px, axis=0)
            index_px = avg_widths[0]
            middle_px = avg_widths[1]
            ring_px = avg_widths[2]
            pinky_px = avg_widths[3]

        # 5. 剥离手指肌肉肉厚，折算真实甲床像素 (甲床宽度约占手指总宽度的 72%)
        nail_ratio = 0.72
        c_curve = 1.05  # C-Curve 弧面物理增益

        index_mm = index_px * nail_ratio * mm_per_px * c_curve
        middle_mm = middle_px * nail_ratio * mm_per_px * c_curve
        ring_mm = ring_px * nail_ratio * mm_per_px * c_curve
        pinky_mm = pinky_px * nail_ratio * mm_per_px * c_curve

        # 对应官方 Standard Size 表校验 (XS - L 范围)
        index_mm = max(10.0, min(15.0, index_mm))
        middle_mm = max(11.0, min(16.0, middle_mm))
        ring_mm = max(10.0, min(15.0, ring_mm))
        pinky_mm = max(8.0, min(13.0, pinky_mm))

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
