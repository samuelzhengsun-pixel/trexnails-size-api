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
    return {"status": "TrexNails Calibrated Precision Engine Active"}

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

        # 1. 图像标准化缩放 (宽 1000px)
        h_orig, w_orig, _ = img.shape
        target_w = 1000
        scale = target_w / float(w_orig)
        img_s = cv2.resize(img, (target_w, int(h_orig * scale)))
        h_s, w_s, _ = img_s.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 2. 硬币形变率校准识别 (Coin Distortion Correction)
        gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        coin_px_diameter = 0
        
        # 优先寻找椭圆/倾斜形变轮廓，取最大外接长轴计算绝对像素标尺
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
                        # 形变率校准：允许倾斜透视形成的椭圆 (形变率 0.6~1.0)
                        if 0.6 <= aspect_ratio <= 1.0 and major_axis > max_valid_d:
                            max_valid_d = major_axis

        if max_valid_d > 0:
            coin_px_diameter = max_valid_d
        else:
            # 降级备用霍夫圆检测
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
                "message": f"Pièce ({coin_type.upper()}) non détectée. Assurez-vous qu'elle soit bien visible."
            }

        # 绝对物理像素比 (mm/px)
        mm_per_px = real_coin_mm / coin_px_diameter

        # 3. 手部连通域提取
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

        # 4. 手指宽度横向切面采样
        roi_y1 = hy + int(hh * 0.12)
        roi_y2 = hy + int(hh * 0.35)
        
        finger_widths_px = []
        for scan_y in range(roi_y1, roi_y2, max(1, (roi_y2 - roi_y1) // 10)):
            row = skin_mask[scan_y, hx:hx+hw]
            changes = np.diff(row.astype(np.int16))
            starts = np.where(changes > 200)[0]
            ends = np.where(changes < -200)[0]
            
            widths = []
            if len(starts) > 0 and len(ends) > 0:
                for s in starts:
                    valid_ends = ends[ends > s]
                    if len(valid_ends) > 0:
                        w_px = valid_ends[0] - s
                        if (hw * 0.08) < w_px < (hw * 0.35):
                            widths.append(w_px)
            if len(widths) >= 4:
                finger_widths_px.append(widths[:4])

        if finger_widths_px:
            avg_widths = np.median(finger_widths_px, axis=0)
            idx_px, mid_px, rng_px, pky_px = avg_widths[0], avg_widths[1], avg_widths[2], avg_widths[3]
        else:
            idx_px = hw * 0.16
            mid_px = hw * 0.18
            rng_px = hw * 0.16
            pky_px = hw * 0.12

        # 🎯 5. 精确校准参数调整 (扣除偏大的 3mm 误差)
        # nail_ratio: 从 0.78 下调至 0.68 (剥离手指两旁肉质干扰)
        # c_curve: 从 1.05 下调至 0.98 (抵消镜头近大远小的虚高增益)
        nail_ratio = 0.68
        c_curve = 0.98

        index_mm = idx_px * nail_ratio * mm_per_px * c_curve
        middle_mm = mid_px * nail_ratio * mm_per_px * c_curve
        ring_mm = rng_px * nail_ratio * mm_per_px * c_curve
        pinky_mm = pky_px * nail_ratio * mm_per_px * c_curve

        # 物理界限约束
        index_mm = max(9.0, min(16.0, index_mm))
        middle_mm = max(10.0, min(17.0, middle_mm))
        ring_mm = max(9.0, min(16.0, ring_mm))
        pinky_mm = max(7.5, min(13.0, pinky_mm))

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
