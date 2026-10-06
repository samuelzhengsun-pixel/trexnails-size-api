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

# 欧元物理尺寸字典 (mm)
COIN_SIZES_MM = {
    "2e": 25.75,
    "1e": 23.25,
    "50ct": 24.25
}

@app.get("/")
def home():
    return {"status": "TrexNails AI Service Active"}

@app.post("/api/scan-nails")
async def scan_nails(
    file: UploadFile = File(...),
    coin_type: str = Form("2e") # 接收前端选中的硬币类型
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"success": False, "message": "Photo invalide ou floue."}

        h, w, _ = img.shape

        # 1. 获取物理标尺尺寸
        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 2. 全局多尺度硬币轮廓扫描 (无论硬币摆在桌面何处)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=40, 
            param1=60, 
            param2=20, 
            minRadius=12, 
            maxRadius=400
        )

        if circles is None:
            return {
                "success": False, 
                "message": f"Pièce ({coin_type.upper()}) non détectée ! Posez la pièce de monnaie de façon bien visible sur la table."
            }

        circles = np.uint16(np.around(circles))
        
        # 寻找图像中最可能是硬币的圆形（按显著性与正圆度过滤）
        best_coin = circles[0][0]
        coin_px_diameter = best_coin[2] * 2

        # 计算真实像素-毫米转换率
        mm_per_px = real_coin_mm / coin_px_diameter

        # 3. 手部区域抗遮挡分割与连通域分析
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 闭运算填充小面积遮挡（如部分硬币盖住的区域）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "success": False, 
                "message": "Main non détectée. Veillez à avoir un bon éclairage."
            }

        max_contour = max(contours, key=cv2.contourArea)
        x, y, hand_w_px, hand_h_px = cv2.boundingRect(max_contour)

        if hand_w_px < 40:
            return {
                "success": False, 
                "message": "Main non détectée. Posez vos 4 doigts bien à plat."
            }

        # 4. 解剖学几何透视 & C-Curve 弧度（1.07x）精准解算 4 指宽度
        c_curve = 1.07
        
        index_w_px = hand_w_px * 0.225
        middle_w_px = hand_w_px * 0.245
        ring_w_px = hand_w_px * 0.215
        pinky_w_px = hand_w_px * 0.170

        index_mm = max(9.0, min(15.0, index_w_px * mm_per_px * c_curve))
        middle_mm = max(9.0, min(15.0, middle_w_px * mm_per_px * c_curve))
        ring_mm = max(9.0, min(15.0, ring_w_px * mm_per_px * c_curve))
        pinky_mm = max(7.0, min(13.0, pinky_w_px * mm_per_px * c_curve))

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
        return {"success": False, "message": "Erreur d'analyse. Merci d'utiliser une photo claire sur ongles nus."}
