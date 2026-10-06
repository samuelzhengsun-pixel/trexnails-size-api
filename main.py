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
    return {"status": "TrexNails AI Service Active"}

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
            return {"success": False, "message": "Photo invalide ou floue."}

        # ⚡ 提速：如图片过大则等比例压缩至 max 1024px 宽，大幅节省传输与 CPU 分析时间
        h, w, _ = img.shape
        if w > 1024:
            scale = 1024.0 / w
            img = cv2.resize(img, (1024, int(h * scale)))
            h, w, _ = img.shape

        real_coin_mm = COIN_SIZES_MM.get(coin_type, 25.75)

        # 1. 硬币轮廓扫描
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=40, 
            param1=60, 
            param2=20, 
            minRadius=10, 
            maxRadius=350
        )

        if circles is None:
            return {
                "success": False, 
                "message": f"Pièce ({coin_type.upper()}) non détectée ! Posez la pièce de monnaie de façon bien visible sur la table."
            }

        circles = np.uint16(np.around(circles))
        best_coin = circles[0][0]
        coin_px_diameter = best_coin[2] * 2

        mm_per_px = real_coin_mm / coin_px_diameter

        # 2. 肤色分割与抗遮挡分析
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
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

        # 📏 解决数据偏大问题：精确缩减物理比例 + 精细 C-Curve (1.03)
        c_curve = 1.03
        
        index_w_px = hand_w_px * 0.205  # 微调原 0.225 -> 0.205
        middle_w_px = hand_w_px * 0.225 # 微调原 0.245 -> 0.225
        ring_w_px = hand_w_px * 0.195   # 微调原 0.215 -> 0.195
        pinky_w_px = hand_w_px * 0.155  # 微调原 0.170 -> 0.155

        index_mm = max(8.5, min(14.5, index_w_px * mm_per_px * c_curve))
        middle_mm = max(8.5, min(14.5, middle_w_px * mm_per_px * c_curve))
        ring_mm = max(8.5, min(14.5, ring_w_px * mm_per_px * c_curve))
        pinky_mm = max(6.5, min(12.5, pinky_w_px * mm_per_px * c_curve))

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
