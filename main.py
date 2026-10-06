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
    return {"status": "TrexNails Orientation-Aware Sizer Active"}

# 旋转图像辅助函数
def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

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

        # 1. 自动方向旋转检测（解决手机竖拍/横拍导致的识别失败）
        best_img = None
        best_results = None
        
        for angle in [0, 90, 270, 180]:
            test_img = rotate_image(img, angle) if angle != 0 else img.copy()
            h_t, w_t, _ = test_img.shape
            if w_t > 1000:
                scale = 1000.0 / w_t
                test_img = cv2.resize(test_img, (1000, int(h_t * scale)))

            img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.1
            ) as hands_detector:
                results = hands_detector.process(img_rgb)
                if results.multi_hand_landmarks:
                    best_img = test_img
                    best_results = results
                    break

        if best_img is None:
            return {
                "success": False, 
                "message": "Main non détectée. Assurez-vous que vos 4 doigts sont bien à plat sur la table."
            }

        img = best_img
        h, w, _ = img.shape
        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 2. 硬币多重保底识别 (霍夫圆 + 轮廓拟合)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        coin_px_diameter = 0

        # 霍夫圆检测
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
            # 边缘轮廓拟合椭圆
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

        mm_per_px = real_coin_mm / coin_px_diameter

        # 3. 抓取指甲 ROI 区域与法线切线
        landmarks = best_results.multi_hand_landmarks[0].landmark

        def get_nail_bed_px(tip_idx, dip_idx):
            tip_x, tip_y = int(landmarks[tip_idx].x * w), int(landmarks[tip_idx].y * h)
            dip_x, dip_y = int(landmarks[dip_idx].x * w), int(landmarks[dip_idx].y * h)

            finger_len_px = math.sqrt((tip_x - dip_x)**2 + (tip_y - dip_y)**2)
            if finger_len_px == 0:
                return 40.0

            margin = int(finger_len_px * 0.5)
            y_min = max(0, min(tip_y, dip_y) - margin)
            y_max = min(h, max(tip_y, dip_y) + margin)
            x_min = max(0, min(tip_x, dip_x) - margin)
            x_max = min(w, max(tip_x, dip_x) + margin)

            roi_gray = gray[y_min:y_max, x_min:x_max]
            if roi_gray.size == 0:
                return finger_len_px * 0.65

            roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
            median_val = np.median(roi_blur)
            edges = cv2.Canny(roi_blur, int(max(0, 0.4 * median_val)), int(min(255, 1.1 * median_val)))

            row_widths = []
            for row in edges:
                pts = np.where(row > 0)[0]
                if len(pts) >= 2:
                    row_widths.append(pts[-1] - pts[0])

            if row_widths:
                nail_px = np.percentile(row_widths, 80)
                if 10 < nail_px < (finger_len_px * 1.3):
                    return nail_px

            return finger_len_px * 0.65

        c_curve = 1.05

        index_px = get_nail_bed_px(8, 7)
        middle_px = get_nail_bed_px(12, 11)
        ring_px = get_nail_bed_px(16, 15)
        pinky_px = get_nail_bed_px(20, 19)

        index_mm = index_px * mm_per_px * c_curve
        middle_mm = middle_px * mm_per_px * c_curve
        ring_mm = ring_px * mm_per_px * c_curve
        pinky_mm = pinky_px * mm_per_px * c_curve

        # 根据官方标准尺码表匹配 (XS - L 范围)
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
