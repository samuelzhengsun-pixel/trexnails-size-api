from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

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
    return {"status": "TrexNails Contour Protection Active"}

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

        # 1. 统一调整图像宽度至 1000px
        h, w, _ = img.shape
        if w > 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 2. 硬币精确抓取 (增加严格面积与圆度校验，剔除桌影噪点)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=50, 
            param1=50, 
            param2=22, 
            minRadius=int(w * 0.05),
            maxRadius=int(w * 0.20)
        )

        coin_px_diameter = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            best_coin = circles[0][0]
            coin_px_diameter = best_coin[2] * 2
        else:
            # 轮廓校验拟合
            edges = cv2.Canny(blurred, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            for c in contours:
                area = cv2.contourArea(c)
                if (w * h * 0.01) < area < (w * h * 0.15):
                    (cx, cy), (d1, d2), _ = cv2.fitEllipse(c)
                    if min(d1, d2) / max(d1, d2) > 0.7 and area > max_area:
                        max_area = area
                        coin_px_diameter = max(d1, d2)

        if coin_px_diameter == 0:
            return {
                "success": False, 
                "message": f"Pièce ({coin_type.upper()}) non détectée. Posez la pièce bien à plat à côté de vos doigts."
            }

        mm_per_px = real_coin_mm / coin_px_diameter

        # 3. HSV 肤色提取与连通域修复 (形态学闭运算消除碎片噪点)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 12, 35], dtype=np.uint8)
        upper_skin = np.array([28, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 核心修复：连通域膨胀与闭运算，填补裂缝
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # 过滤面积过小的杂质噪点，确保只识别整块手部
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.05)]

        if not valid_contours:
            return {"success": False, "message": "Main non détectée. Veillez à poser vos doigts sur un fond clair."}

        hand_contour = max(valid_contours, key=cv2.contourArea)
        x, y, hand_w, hand_h = cv2.boundingRect(hand_contour)

        # 校验手部物理有效跨度（防止抓取到非手部大面积背景）
        if hand_w < (w * 0.15) or hand_h < (h * 0.15):
            return {"success": False, "message": "Main non détectée correctement. Veuillez reprendre la photo."}

        # 4. 横向指甲最宽处像素采样
        roi_y1 = y + int(hand_h * 0.08)
        roi_y2 = y + int(hand_h * 0.35)
        
        finger_widths_px = []
        for scan_y in range(roi_y1, roi_y2, max(1, (roi_y2 - roi_y1) // 10)):
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
                        if (hand_w * 0.10) < w_px < (hand_w * 0.38):
                            widths.append(w_px)
            if len(widths) >= 4:
                finger_widths_px.append(widths[:4])

        if not finger_widths_px:
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

        # 5. 1:1 绝对物理解算映射 (对应标准尺寸 14, 15, 14, 12 mm)
        nail_to_finger_ratio = 0.75
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
