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

# 严格遵循欧盟官方硬币物理直径 (绝不颠倒)
COIN_SIZES_MM = {
    "2e": 25.75,    # 2€ 25.75mm
    "50ct": 24.25,  # 50ct 24.25mm
    "1e": 23.25     # 1€ 23.25mm
}

@app.get("/")
def home():
    return {"status": "TrexNails Strict Physical Calibration Engine Active"}

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

        # 1. 严格清洗 coin_type，确保 1e, 50ct, 2e 准确映射物理毫米数
        coin_key = str(coin_type).strip().lower()
        if coin_key not in COIN_SIZES_MM:
            coin_key = "2e"
            
        real_coin_mm = COIN_SIZES_MM[coin_key]

        # 2. 图像标准化缩放 (宽 1000px)
        h_orig, w_orig, _ = img.shape
        target_w = 1000
        scale = target_w / float(w_orig)
        img_s = cv2.resize(img, (target_w, int(h_orig * scale)))
        h_s, w_s, _ = img_s.shape
        
        img_center_x, img_center_y = w_s / 2.0, h_s / 2.0
        max_dist_to_center = math.sqrt(img_center_x**2 + img_center_y**2)

        # 3. 硬币标尺检测 (椭圆拟合长轴)
        gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        coin_px_diameter = 0
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_valid_d = 0
        for c in contours:
            area = cv2.contourArea(c)
            if (w_s * h_s * 0.003) < area < (w_s * h_s * 0.15):
                if len(c) >= 5:
                    (cx, cy), (d1, d2), angle = cv2.fitEllipse(c)
                    major_axis = max(d1, d2)
                    minor_axis = min(d1, d2)
                    if minor_axis > 0:
                        aspect_ratio = minor_axis / major_axis
                        if 0.6 <= aspect_ratio <= 1.0 and major_axis > max_valid_d:
                            max_valid_d = major_axis

        if max_valid_d > 0:
            coin_px_diameter = max_valid_d
        else:
            for param2 in [25, 20, 15]:
                circles = cv2.HoughCircles(
                    blurred, 
                    cv2.HOUGH_GRADIENT, 
                    dp=1.2, 
                    minDist=50, 
                    param1=50, 
                    param2=param2, 
                    minRadius=int(w_s * 0.03), 
                    maxRadius=int(w_s * 0.18)
                )
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    best_c = circles[0][0]
                    coin_px_diameter = best_c[2] * 2
                    break

        if coin_px_diameter == 0:
            return {
                "success": False, 
                "message": f"Pièce ({coin_key.upper()}) non détectée. Assurez-vous qu'elle soit bien visible."
            }

        # 物理像素比例尺 (绝对遵循 real_coin_mm / coin_px_diameter)
        mm_per_px = real_coin_mm / float(coin_px_diameter)

        # 4. 手部 Mask 提取与采样
        hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 15, 30], dtype=np.uint8)
        upper_skin = np.array([28, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > (w_s * h_s * 0.04)]

        if not valid_contours:
            return {"success": False, "message": "Main non détectée. Posez vos doigts sur un fond clair."}

        hand_contour = max(valid_contours, key=cv2.contourArea)
        hx, hy, hw, hh = cv2.boundingRect(hand_contour)

        roi_y1 = hy + int(hh * 0.12)
        roi_y2 = hy + int(hh * 0.35)
        
        finger_raw_widths = []
        finger_centers_x = []
        finger_centers_y = []

        for scan_y in range(roi_y1, roi_y2, max(1, (roi_y2 - roi_y1) // 10)):
            row_skin = skin_mask[scan_y, hx:hx+hw]
            changes = np.diff(row_skin.astype(np.int16))
            starts = np.where(changes > 200)[0]
            ends = np.where(changes < -200)[0]
            
            widths, cx_list, cy_list = [], [], []

            if len(starts) > 0 and len(ends) > 0:
                for s in starts:
                    valid_ends = ends[ends > s]
                    if len(valid_ends) > 0:
                        e = valid_ends[0]
                        total_finger_px = e - s
                        if (hw * 0.08) < total_finger_px < (hw * 0.35):
                            widths.append(total_finger_px)
                            cx_list.append(hx + s + total_finger_px / 2.0)
                            cy_list.append(scan_y)

            if len(widths) >= 4:
                finger_raw_widths.append(widths[:4])
                finger_centers_x.append(cx_list[:4])
                finger_centers_y.append(cy_list[:4])

        if finger_raw_widths:
            avg_widths = np.median(finger_raw_widths, axis=0)
            avg_cx = np.median(finger_centers_x, axis=0)
            avg_cy = np.median(finger_centers_y, axis=0)
        else:
            avg_widths = [hw * 0.15, hw * 0.17, hw * 0.15, hw * 0.11]
            avg_cx = [hx + hw * 0.2, hx + hw * 0.4, hx + hw * 0.6, hx + hw * 0.8]
            avg_cy = [hy + hh * 0.2, hy + hh * 0.15, hy + hh * 0.2, hy + hh * 0.28]

        # 5. 保持完全不变的精化解剖锚定计算 (保持小拇指与前三指的精准比例)
        pinky_y = avg_cy[3]
        pinky_x = avg_cx[3]
        
        base_nail_ratio = 0.58
        c_curve = 0.95

        calibrated_mm_list = []

        for i in range(4):
            raw_w_px = avg_widths[i]
            fx, fy = avg_cx[i], avg_cy[i]

            if i == 3:
                w_rel = 1.03
            else:
                y_diff = max(0.0, pinky_y - fy)
                rel_height_ratio = y_diff / max(1.0, hh * 0.25)
                
                dist_to_center = math.sqrt((fx - img_center_x)**2 + (fy - img_center_y)**2)
                pinky_dist_to_center = math.sqrt((pinky_x - img_center_x)**2 + (pinky_y - img_center_y)**2)
                rel_center_ratio = max(0.0, (pinky_dist_to_center - dist_to_center) / max_dist_to_center)

                w_rel = 1.0 - (rel_height_ratio * 0.05) - (rel_center_ratio * 0.03)
                w_rel = max(0.91, min(1.0, w_rel))

            final_mm = raw_w_px * base_nail_ratio * mm_per_px * c_curve * w_rel
            calibrated_mm_list.append(final_mm)

        index_mm  = max(8.5, min(15.0, calibrated_mm_list[0]))
        middle_mm = max(9.5, min(16.0, calibrated_mm_list[1]))
        ring_mm   = max(8.5, min(15.0, calibrated_mm_list[2]))
        pinky_mm  = max(7.0, min(12.5, calibrated_mm_list[3]))

        return {
            "success": True,
            "coin_used": coin_key.upper(),
            "coin_mm": real_coin_mm,
            "measures": {
                "index": round(index_mm, 1),
                "middle": round(middle_mm, 1),
                "ring": round(ring_mm, 1),
                "pinky": round(pinky_mm, 1)
            }
        }
    except Exception as e:
        return {"success": False, "message": "Erreur d'analyse photo."}
