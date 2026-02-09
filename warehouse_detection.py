import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
from collections import deque
from pyzbar.pyzbar import decode as decode_barcode

# --- AYARLAR ---
FRAME_SOURCE   = "shared_folder"
SHARED_FOLDER  = r"C:\Users\simuser\Desktop\WareDrone-Meshine\sim_images\sim"
ARUCO_DICT_ID  = cv2.aruco.DICT_6X6_250
WINDOW_SIZE    = None # Örn: (1280, 720)

# Simülasyon Kamerasının Tahmini Fiziksel Parametreleri (Kalibrasyon yoksa)
# Isaac Sim'deki kameranın Horizontal FOV değeri genelde 60 veya 90 derecedir.
SIM_FOV_DEGREE = 60.0 
REAL_MARKER_WIDTH_M = 0.20  # Marker'ın gerçek hayattaki boyutu (metre cinsinden, örn: 20cm)

MAX_HISTORY_SIZE = 200
BARCODE_QUALITY_THRESHOLD = 0

# Renkler (BGR)
COLOR_BARCODE  = (0, 255, 0)
COLOR_ARUCO    = (0, 180, 255)
COLOR_TEXT     = (255, 255, 255)
COLOR_BG       = (0, 0, 0)
COLOR_LOW_CONF = (100, 100, 255)


class SharedFolderReader:
    def __init__(self, folder_path: str):
        self.folder = Path(folder_path)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.last_mtime = 0
        self._last_frame = None
        self.wait_count = 0
        self.file_path = self.folder / "latest_frame.png"

    def get_frame(self):
        if not self.file_path.exists():
            if self.wait_count % 60 == 0: # Konsolu spamlamamak için seyrek log
                print(f"[WAIT] {self.file_path} bekleniyor... Isaac Sim çalışıyor mu?")
            self.wait_count += 1
            return self._placeholder_frame(), False

        try:
            # 1. Optimasyon: Dosya içeriğini okumadan önce metadata'ya bak (Hızlı)
            stat = self.file_path.stat()
            current_mtime = stat.st_mtime_ns

            if current_mtime == self.last_mtime:
                # Dosya değişmemiş, eski frame'i döndür (veya None)
                return self._last_frame, True

            # 2. Race Condition Koruması: Dosya yazılırken okumayı önlemek için basit retry
            # Simülasyon dosyayı yazarken kilitli olabilir veya bozuk veri gelebilir.
            frame = None
            for _ in range(3): 
                try:
                    # cv2.imread dosya bozuksa None döner, exception fırlatmaz.
                    temp_frame = cv2.imread(str(self.file_path))
                    if temp_frame is not None and temp_frame.size > 0:
                        frame = temp_frame
                        break
                except Exception:
                    time.sleep(0.01)

            if frame is not None:
                self.last_mtime = current_mtime
                self._last_frame = frame
                return frame, True
            
            return self._last_frame, True # Okuyamadıysak son geçerli frame'i göster

        except OSError:
            return None, False

    def _placeholder_frame(self):
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "WAITING FOR SIM...", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return placeholder


class WebcamReader:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Webcam açılamadı!")
            sys.exit(1)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame, ret

    def release(self):
        self.cap.release()


class DetectionEngine:
    def __init__(self, aruco_dict_id: int):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # ArUco parametre optimizasyonu
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params
        )

        # Fiziksel hesaplama sabitleri (Daha sonra frame genişliğine göre güncellenecek)
        self.focal_length_px = None 

    def _calculate_focal_length(self, image_width):
        """
        Pinhole kamera modeli için Focal Length (piksel cinsinden) tahmini.
        F = (W / 2) / tan(FOV / 2)
        """
        if self.focal_length_px is None:
            fov_rad = np.deg2rad(SIM_FOV_DEGREE)
            self.focal_length_px = (image_width / 2) / np.tan(fov_rad / 2)
            print(f"[INFO] Tahmini Focal Length: {self.focal_length_px:.2f}px (FOV: {SIM_FOV_DEGREE}°)")

    def detect(self, frame: np.ndarray) -> dict:
        self._calculate_focal_length(frame.shape[1])
        
        results = {"barcodes": [], "arucos": []}
        
        # 1. Barkod Tespiti
        # Renkli görüntüde barkod okunabilir ama gray daha hızlıdır
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        barcodes = decode_barcode(gray)
        for bc in barcodes:
            if bc.quality < BARCODE_QUALITY_THRESHOLD:
                continue
            
            pts = np.array(bc.polygon, dtype=np.int32).reshape(-1, 1, 2)
            results["barcodes"].append({
                "data": bc.data.decode("utf-8", errors="ignore"),
                "points": pts,
                "quality": bc.quality
            })

        # 2. ArUco Tespiti
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i]
                distance_est = self._estimate_distance_pinhole(marker_corners)
                
                results["arucos"].append({
                    "id": int(marker_id),
                    "corners": marker_corners,
                    "distance": distance_est
                })

        return results

    def _estimate_distance_pinhole(self, corners: np.ndarray) -> float:
        """
        Üçgen benzerliği kullanarak mesafe hesabı:
        D = (RealWidth * FocalLength) / PixelWidth
        """
        # Köşeler: [TopLeft, TopRight, BottomRight, BottomLeft]
        p0, p1 = corners[0][0], corners[0][1]
        
        # Marker'ın görüntüdeki genişliği (Piksel)
        width_px = np.linalg.norm(p1 - p0)
        
        if width_px > 0 and self.focal_length_px:
            distance = (REAL_MARKER_WIDTH_M * self.focal_length_px) / width_px
        else:
            distance = 0.0
        
        return distance


class Renderer:
    def __init__(self, max_history: int = MAX_HISTORY_SIZE):
        self.unique_barcodes = set()
        self.unique_arucos = set()
        self.total_unique_detections = 0

    def draw(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        output = frame.copy()
        
        # Barkod Çizimi
        for bc in detections["barcodes"]:
            pts = bc["points"]
            color = COLOR_BARCODE
            
            cv2.polylines(output, [pts], True, color, 2)
            
            x, y = pts[0][0]
            label = f"BC: {bc['data']}"
            self._draw_label(output, label, (x, y - 8), color)
            
            if bc["data"] not in self.unique_barcodes:
                self.unique_barcodes.add(bc["data"])
                self.total_unique_detections += 1

        # ArUco Çizimi
        for ac in detections["arucos"]:
            corners = ac["corners"].reshape(-1, 1, 2).astype(np.int32)
            distance = ac.get("distance", 0)
            
            cv2.polylines(output, [corners], True, COLOR_ARUCO, 2)
            
            x, y = corners[0][0]
            label = f"ID: {ac['id']} | {distance:.2f}m"
            self._draw_label(output, label, (x, y - 8), COLOR_ARUCO)
            
            if ac["id"] not in self.unique_arucos:
                self.unique_arucos.add(ac["id"])
                self.total_unique_detections += 1

        self._draw_stats(output, detections)
        return output

    def _draw_label(self, frame, text, pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        x, y = pos
        # Arka plan kutusu (okunabilirlik için)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw, y + baseline), COLOR_BG, -1)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def _draw_stats(self, frame, detections):
        h, w = frame.shape[:2]
        stats = [
            f"FPS: {detections.get('fps', 0):.1f}",
            f"Active BC: {len(detections['barcodes'])}",
            f"Active ArUco: {len(detections['arucos'])}",
            f"Total Unique: {self.total_unique_detections}"
        ]
        
        for i, text in enumerate(stats):
            cv2.putText(frame, text, (10, 30 + (i * 25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def reset(self):
        self.unique_barcodes.clear()
        self.unique_arucos.clear()
        self.total_unique_detections = 0


def main():
    if FRAME_SOURCE == "shared_folder":
        reader = SharedFolderReader(SHARED_FOLDER)
    else:
        reader = WebcamReader()

    engine = DetectionEngine(ARUCO_DICT_ID)
    renderer = Renderer()

    window_name = "WareDrone Logic Fix"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if WINDOW_SIZE:
        cv2.resizeWindow(window_name, WINDOW_SIZE[0], WINDOW_SIZE[1])

    fps_start = time.time()
    frames_processed = 0
    current_fps = 0

    print("[INFO] Loop başladı...")

    while True:
        frame, is_new = reader.get_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        # Sadece yeni frame geldiyse veya webcam ise işle
        # Shared folder'da aynı frame'i tekrar tekrar işlemeye gerek yok (CPU tasarrufu)
        if is_new or FRAME_SOURCE == "webcam":
            detections = engine.detect(frame)
            detections['fps'] = current_fps # FPS bilgisini renderera pasla
            output = renderer.draw(frame, detections)
            
            # FPS Hesabı
            frames_processed += 1
            if frames_processed % 30 == 0:
                elapsed = time.time() - fps_start
                current_fps = 30 / elapsed if elapsed > 0 else 0
                fps_start = time.time()
            
            cv2.imshow(window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            renderer.reset()
            print("[INFO] Resetlendi.")

    cv2.destroyAllWindows()
    if hasattr(reader, 'release'):
        reader.release()

if __name__ == "__main__":
    main()