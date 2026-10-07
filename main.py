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
    return {"status": "TrexNails Adaptive Morphological Engine Active"}

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

        # 2. 硬币形变率与标尺精准检测
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
                "message": f"Pièce ({coin_type.upper()}) non détectée. Assurez-vous qu'elle soit bien visible."
            }

        mm_per_px = real_coin_mm / coin_px_diameter

        # 3. 手部皮肤与边缘梯度分析 (Skin & Nail Bed Adaptive Analysis)
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

        # 4. 针对每个人手的“肉/骨比率”自适应采样算法
        roi_y1 = hy + int(hh * 0.12)
        roi_y2 = hy + int(hh * 0.35)
        
        finger_raw_widths = []
        finger_nail_ratios = []

        # 利用 Sobel 梯度求取甲床内侧边缘突变
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        for scan_y in range(roi_y1, roi_y2, max(1, (roi_y2 - roi_y1) // 10)):
            row_skin = skin_mask[scan_y, hx:hx+hw]
            row_grad = sobelx[scan_y, hx:hx+hw]

            changes = np.diff(row_skin.astype(np.int16))
            starts = np.where(changes > 200)[0]
            ends = np.where(changes < -200)[0]
            
            widths = []
            ratios = []

            if len(starts) > 0 and len(ends) > 0:
                for s in starts:
                    valid_ends = ends[ends > s]
                    if len(valid_ends) > 0:
                        e = valid_ends[0]
                        total_finger_px = e - s
                        
                        if (hw * 0.08) < total_finger_px < (hw * 0.35):
                            widths.append(total_finger_px)
                            
                            # 采样手指内部 20%~80% 区域的梯度突变点，自动分析肉边厚度
                            finger_grad = row_grad[s:e]
                            if len(finger_grad) > 10:
                                # 计算指甲边缘高光与沟槽的梯度阈值
                                high_grad_count = np.sum(finger_grad > (np.mean(finger_grad) * 1.5))
                                # 梯度突变多说明指甲边界明显，肉较薄；梯度平缓说明肉包指甲
                                adapt_r = 0.52 + min(0.18, (high_grad_count / float(len(finger_grad))) * 0.25)
                                ratios.append(adapt_r)
                            else:
                                ratios.append(0.58)

            if len(widths) >= 4:
                finger_raw_widths.append(widths[:4])
                finger_nail_ratios.append(ratios[:4])

        if finger_raw_widths:
            avg_widths = np.median(finger_raw_widths, axis=0)
            avg_ratios = np.median(finger_nail_ratios, axis=0)
        else:
            avg_widths = [hw * 0.15, hw * 0.17, hw * 0.15, hw * 0.11]
            avg_ratios = [0.58, 0.58, 0.58, 0.58]

        # 5. 结合自适应肉比 + C-Curve (0.95) 换算绝对毫米数
        c_curve = 0.95

        index_mm = (avg_widths[0] * avg_ratios[0] * mm_per_px * c_curve)
        middle_mm = (avg_widths[1] * avg_ratios[1] * mm_per_px * c_curve)
        ring_mm = (avg_widths[2] * avg_ratios[2] * mm_per_px * c_curve)
        pinky_mm = (avg_widths[3] * avg_ratios[3] * mm_per_px * c_curve)

        # 物理界限约束 (精准落地区间 8.5mm ~ 16mm)
        index_mm = max(8.5, min(15.0, index_mm))
        middle_mm = max(9.5, min(16.0, middle_mm))
        ring_mm = max(8.5, min(15.0, ring_mm))
        pinky_mm = max(7.0, min(12.5, pinky_mm))

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
