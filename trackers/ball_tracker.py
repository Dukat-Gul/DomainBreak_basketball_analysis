import pickle
import os
from ultralytics import YOLO

# Corretto il nome della classe importata
from utils.kalman_filter import DetectionsToTracksKalmanFilter


class BallTracker:
    def __init__(
        self,
        model,
        imgsz=960,
        min_conf=0.05,
        max_search_radius=0.10,
        area_ratio_min=2e-5,
        area_ratio_max=5e-3,
        use_tta=False,
        w_conf=1.0,
        w_orange=0.35,
        w_dist=0.2,
    ):
        self.model = model
        # Parametri per migliorare recall della palla
        self.imgsz = imgsz
        self.min_conf = min_conf
        # Raggio di ricerca relativo (in frazione del lato massimo) attorno all'ultima posizione
        self.max_search_radius = max_search_radius
        # Priors sulle dimensioni relative della palla (area bbox rispetto all'area del frame)
        self.area_ratio_min = area_ratio_min
        self.area_ratio_max = area_ratio_max
        # Test-time augmentation (flip/scale) per aumentare il recall
        self.use_tta = use_tta
        # Pesi per scoring dei candidati
        self.w_conf = w_conf
        self.w_orange = w_orange
        self.w_dist = w_dist

        # Identifica dinamicamente l'ID della classe "ball" dal modello
        try:
            self.ball_class_id = next(
                k for k, v in self.model.names.items() if v.lower() == "ball"
            )
        except StopIteration:
            # Fallback: se il modello è single-class, di solito 'ball' è id 0
            self.ball_class_id = 0

        self.tracker = DetectionsToTracksKalmanFilter()
        self.tracks = {}
        self.last_bbox = None  # ultima bbox confermata

    def _orange_ratio(self, frame, bbox):
        import cv2
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Fascia arancione approssimata in HSV
        lower = (5, 70, 40)
        upper = (25, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        return float(mask.mean() / 255.0)

    def detect_frames(self, frames):
        detections = []
        for frame in frames:
            # Filtra direttamente per la classe 'ball'
            ball_detections = self.model.predict(
                frame,
                conf=self.min_conf,
                imgsz=self.imgsz,
                classes=[self.ball_class_id],
                augment=self.use_tta,
            )
            chosen = []
            if len(ball_detections[0].boxes) > 0:
                boxes = ball_detections[0].boxes
                h, w = frame.shape[:2]
                frame_area = float(h * w)
                # Pre-calcolo per distanza
                if self.last_bbox is not None:
                    px1, py1, px2, py2 = self.last_bbox
                    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                    r2 = (self.max_search_radius * max(w, h)) ** 2
                else:
                    pcx = pcy = None
                    r2 = None

                def score_box(b):
                    x1, y1, x2, y2 = b.xyxy.squeeze().tolist()
                    conf = float(b.conf)
                    area_ratio = max(1.0, (x2 - x1) * (y2 - y1)) / frame_area
                    # Penalità se fuori range area
                    if area_ratio < self.area_ratio_min or area_ratio > self.area_ratio_max:
                        area_pen = 0.5
                    else:
                        area_pen = 1.0
                    # Distanza normalizzata (se possibile)
                    if pcx is not None and r2 and r2 > 0:
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        d2 = (cx - pcx) ** 2 + (cy - pcy) ** 2
                        dist_score = max(0.0, 1.0 - (d2 / r2))
                    else:
                        dist_score = 0.0
                    # Colore arancione
                    orange = self._orange_ratio(frame, (x1, y1, x2, y2))
                    # Score complessivo
                    return (
                        self.w_conf * conf + self.w_orange * orange + self.w_dist * dist_score
                    ) * area_pen

                # Se esiste una posizione precedente, applichiamo gating duro: consideriamo solo box entro raggio
                if pcx is not None:
                    def within_gate(b):
                        x1, y1, x2, y2 = b.xyxy.squeeze().tolist()
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        d2 = (cx - pcx) ** 2 + (cy - pcy) ** 2
                        return d2 <= r2

                    gated = [b for b in boxes if within_gate(b)]
                    candidate_set = gated if gated else list(boxes)
                else:
                    candidate_set = list(boxes)

                if candidate_set:
                    chosen_det = max(candidate_set, key=score_box)
                    chosen.append(chosen_det)

            detections.append(chosen)
        return detections

    def update_tracks(self, detections):
        for frame_num, detection in enumerate(detections):
            if not detection:
                self.tracks[frame_num] = {}
                continue

            bbox = detection[0].xyxy.squeeze().tolist()
            conf = float(detection[0].conf.squeeze().tolist())

            # Soppressione dimensionale per evitare FP (teste/spalle):
            # limita il cambio di scala rispetto al bbox precedente
            if self.last_bbox is not None:
                prev_w = self.last_bbox[2] - self.last_bbox[0]
                prev_h = self.last_bbox[3] - self.last_bbox[1]
                prev_area = max(1.0, prev_w * prev_h)
                w = max(1.0, bbox[2] - bbox[0])
                h = max(1.0, bbox[3] - bbox[1])
                area = w * h
                if area > 3.0 * prev_area or area < 0.2 * prev_area:
                    # Tratta come miss per evitare salti improbabili
                    tracked_bbox = self.tracker.process_detections(None, conf)
                else:
                    tracked_bbox = self.tracker.process_detections(bbox, conf)
            else:
                tracked_bbox = self.tracker.process_detections(bbox, conf)

            if tracked_bbox:
                # Aggiungiamo anche la class_id per la valutazione
                self.tracks[frame_num] = {
                    1: {"bbox": tracked_bbox, "score": conf, "class_id": self.ball_class_id}
                }
                self.last_bbox = tracked_bbox
            else:
                self.tracks[frame_num] = {}

    def track_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Caricamento tracce della palla da stub: {stub_path}")
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        print(
            "Esecuzione del tracking della palla (nessuno stub trovato o richiesto)..."
        )
        detections = self.detect_frames(frames)
        self.update_tracks(detections)

        if stub_path:
            print(f"Salvataggio tracce della palla in stub: {stub_path}")
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(self.tracks, f)

        return self.tracks
