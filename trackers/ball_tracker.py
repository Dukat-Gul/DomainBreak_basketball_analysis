import pickle
import os
from ultralytics import YOLO
import math
import numpy as np

# Corretto il nome della classe importata
from utils.kalman_filter import DetectionsToTracksKalmanFilter


class BallTracker:
    def __init__(
        self,
        model,
        imgsz=960,
        min_conf=0.05,
        iou_nms=0.85,
        max_det=500,
        agnostic_nms=False,
        max_search_radius=0.10,
        r_min=0.06,
        r_max=0.20,
        r_k=0.05,
        area_ratio_min=2e-5,
        area_ratio_max=5e-3,
        use_tta=False,
        w_conf=1.0,
        w_orange=0.35,
        w_dist=0.2,
        w_shape=0.2,
        w_maha=0.4,
        chi2_gating=9.21,
        debug_ball=False,
        select_by_conf=False,
        kalman_max_misses=8,
    ):
        self.model = model
        # Parametri per migliorare recall della palla
        self.imgsz = imgsz
        self.min_conf = min_conf
        self.iou_nms = iou_nms
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        # Raggio di ricerca relativo (in frazione del lato massimo) attorno all'ultima posizione
        self.max_search_radius = max_search_radius
        self.r_min = r_min
        self.r_max = r_max
        self.r_k = r_k
        # Priors sulle dimensioni relative della palla (area bbox rispetto all'area del frame)
        self.area_ratio_min = area_ratio_min
        self.area_ratio_max = area_ratio_max
        # Test-time augmentation (flip/scale) per aumentare il recall
        self.use_tta = use_tta
        # Pesi per scoring dei candidati
        self.w_conf = w_conf
        self.w_orange = w_orange
        self.w_dist = w_dist
        self.w_shape = w_shape
        self.w_maha = w_maha
        self.chi2_gating = chi2_gating
        self.debug_ball = debug_ball
        self.select_by_conf = bool(select_by_conf)
        self.prev_patch = None

        # --- ROI re-detect ---
        self.roi_scale = 2.4  # fattore di ingrandimento ROI attorno all'ultima bbox
        self.roi_limit = 1280  # imgsz massimo SOLO per la ROI
        self.roi_conf_factor = 0.6  # abbassa conf nella ROI: min_conf * factor

        # --- Tile re-detect (multi-scala per oggetti piccoli) ---
        self.tile_grid = (2, 2)  # griglia tasselli (cols, rows)
        self.tile_overlap = 0.25  # overlap fra tasselli (25%)
        self.tile_imgsz = 960  # imgsz per ogni tassello (alta risoluzione)
        self.tile_conf = 0.03  # conf minima sui tiles
        self.tile_max_det = 1000  # allow many candidates per tassello

        # Identifica dinamicamente l'ID della classe "ball" dal modello
        try:
            self.ball_class_id = next(
                k for k, v in self.model.names.items() if v.lower() == "ball"
            )
        except StopIteration:
            # Fallback: se il modello è single-class, di solito 'ball' è id 0
            self.ball_class_id = 0

        self.tracker = DetectionsToTracksKalmanFilter(max_misses=int(kalman_max_misses))
        self.tracks = {}
        self.last_bbox = None  # ultima bbox confermata
        self.tm_patch_size = 31
        self.tm_thresh = 0.60

        # --- Adaptive search/thresholds (Step 1) ---
        self.adaptive = True  # abilita/disabilita la logica adattiva
        self.min_conf_floor = (
            0.02  # pavimento di conf per aumentare il recall in recovery
        )
        self.chi2_relax_max = 16.0  # massimo gating ellittico quando la palla è persa
        self.imgsz_boost = max(1024, int(self.imgsz))  # imgsz usato durante il recupero
        # ROI scaling dinamico
        self.roi_scale_max = 3.5
        # Tiling dinamico
        self.tile_grid_alt = (3, 3)
        self.tile_overlap_alt = 0.35

        # --- Metrics logging ---
        self.metrics = []

        # --- Player proximity prior (Step 2) ---
        # Identifica classi giocatore (nomi contenenti 'player' o 'person')
        self.player_class_ids = set(
            k
            for k, v in getattr(self.model, "names", {}).items()
            if isinstance(v, str) and ("player" in v.lower() or "person" in v.lower())
        )
        self.enable_player_proximity = True
        self.player_min_conf = 0.25
        self.player_max_det = 200
        self.w_player_prox = 0.15  # peso bonus prossimità ai giocatori
        self.prox_r_frac = 0.12  # raggio prossimità rispetto al lato max del frame

        # --- Motion/IoU coherence (Step 3)
        self.w_iou = 0.20  # peso bonus dell'IoU con la bbox precedente
        self.min_fused_select = (
            0.0  # soglia minima per accettare un candidato (0 = disabilitato)
        )

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

    def _conf_sigmoid(self, c: float) -> float:
        return 1.0 / (1.0 + math.exp(-(float(c) - 0.2) / 0.1))

    def _tm_fallback(self, frame, pred_center, search_radius_px):
        import cv2

        if self.prev_patch is None or pred_center is None:
            return None, 0.0
        h, w = frame.shape[:2]
        cx, cy = map(int, pred_center)
        r = int(search_radius_px)
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        roi = frame[y1:y2, x1:x2]
        ph, pw = self.prev_patch.shape[:2]
        if roi.shape[0] < ph or roi.shape[1] < pw:
            return None, 0.0
        res = cv2.matchTemplate(roi, self.prev_patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= self.tm_thresh:
            top_left = (x1 + max_loc[0], y1 + max_loc[1])
            center = (top_left[0] + pw // 2, top_left[1] + ph // 2)
            if self.last_bbox is not None:
                lw = self.last_bbox[2] - self.last_bbox[0]
                lh = self.last_bbox[3] - self.last_bbox[1]
                return [
                    center[0] - lw / 2,
                    center[1] - lh / 2,
                    center[0] + lw / 2,
                    center[1] + lh / 2,
                ], float(max_val)
        return None, float(max_val)

    def _player_proximity_score(self, bbox, players, frame_w, frame_h):
        """
        Restituisce un punteggio [0,1] maggiore quando la bbox è vicina a un giocatore.
        Definisce un raggio di prossimità in pixel come prox_r_frac * max(w,h).
        """
        if not self.enable_player_proximity or not players:
            return 0.0
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        r = float(self.prox_r_frac) * max(frame_w, frame_h)
        if r <= 1e-3:
            return 0.0
        best = 0.0
        for px1, py1, px2, py2 in players:
            pcx = (px1 + px2) * 0.5
            pcy = (py1 + py2) * 0.5
            d = math.hypot(cx - pcx, cy - pcy)
            score = max(0.0, 1.0 - (d / r))
            if score > best:
                best = score
        return float(best)

    def _effective_params(self):
        """Parametri efficaci per il frame corrente basati su self.tracker.misses."""
        misses = int(getattr(self.tracker, "misses", 0))
        if not self.adaptive or misses <= 0:
            return {
                "min_conf": self.min_conf,
                "imgsz": self.imgsz,
                "chi2": self.chi2_gating,
                "roi_scale": self.roi_scale,
                "roi_conf_factor": self.roi_conf_factor,
                "tile_grid": self.tile_grid,
                "tile_overlap": self.tile_overlap,
                "tile_imgsz": self.tile_imgsz,
                "tile_conf": self.tile_conf,
            }

        if misses <= 2:
            # Lieve rilassamento
            eff_min_conf = max(self.min_conf_floor, float(self.min_conf) * 0.7)
            eff_imgsz = max(int(self.imgsz), int(self.imgsz))
            eff_chi2 = min(self.chi2_relax_max, float(self.chi2_gating) + 4.0)
            eff_roi_scale = min(self.roi_scale_max, float(self.roi_scale) * 1.25)
            eff_tile_grid = self.tile_grid
            eff_tile_overlap = self.tile_overlap
            eff_tile_imgsz = self.tile_imgsz
            eff_tile_conf = self.tile_conf
        else:
            # Rilassamento deciso
            eff_min_conf = max(self.min_conf_floor, float(self.min_conf) * 0.5)
            eff_imgsz = max(int(self.imgsz), int(self.imgsz_boost))
            eff_chi2 = float(self.chi2_relax_max)
            eff_roi_scale = min(self.roi_scale_max, float(self.roi_scale) * 1.5)
            eff_tile_grid = self.tile_grid_alt
            eff_tile_overlap = self.tile_overlap_alt
            eff_tile_imgsz = max(self.tile_imgsz, eff_imgsz)
            eff_tile_conf = max(0.02, float(self.tile_conf) * 0.8)

        return {
            "min_conf": eff_min_conf,
            "imgsz": eff_imgsz,
            "chi2": eff_chi2,
            "roi_scale": eff_roi_scale,
            "roi_conf_factor": self.roi_conf_factor,
            "tile_grid": eff_tile_grid,
            "tile_overlap": eff_tile_overlap,
            "tile_imgsz": eff_tile_imgsz,
            "tile_conf": eff_tile_conf,
        }

    def _roi_redetect(
        self, frame, ref_bbox, conf=None, scale=None, roi_limit=None, players=None
    ):
        """
        Ritenta la detection SOLO dentro una ROI centrata sull'ultima bbox nota,
        ingrandita di 'scale'. Esegue YOLO sulla ROI con imgsz alto (fino a roi_limit).
        Restituisce un dict 'best_item' come in detect_frames(), oppure None.
        """
        import cv2, math

        x1, y1, x2, y2 = [int(v) for v in ref_bbox]
        h, w = frame.shape[:2]
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        bw, bh = (x2 - x1), (y2 - y1)

        scale = self.roi_scale if scale is None else scale
        conf = max(
            0.02, self.min_conf * (self.roi_conf_factor if conf is None else conf)
        )
        roi_limit = self.roi_limit if roi_limit is None else roi_limit

        # ROI ingrandita e clamp ai bordi
        rw = int(max(12, bw * scale))
        rh = int(max(12, bh * scale))
        rx1 = max(0, int(cx - rw * 0.5))
        ry1 = max(0, int(cy - rh * 0.5))
        rx2 = min(w, int(cx + rw * 0.5))
        ry2 = min(h, int(cy + rh * 0.5))
        if rx2 - rx1 < 6 or ry2 - ry1 < 6:
            return None

        roi = frame[ry1:ry2, rx1:rx2]
        imgsz_roi = min(roi_limit, max(roi.shape[0], roi.shape[1]))

        preds = self.model.predict(
            roi,
            conf=conf,
            imgsz=imgsz_roi,
            classes=[self.ball_class_id],
            augment=self.use_tta,
            verbose=False,
        )
        if len(preds[0].boxes) == 0:
            return None

        best_item, best_score = None, -1e9
        hF, wF = frame.shape[:2]
        frame_area = float(hF * wF)

        def conf_sigmoid(c):  # stesso shaping usato nel main
            return 1.0 / (1.0 + math.exp(-(float(c) - 0.2) / 0.1))

        for b in preds[0].boxes:
            x1r, y1r, x2r, y2r = b.xyxy.squeeze().tolist()
            # rimappa coordinate ROI -> frame
            X1 = rx1 + x1r
            Y1 = ry1 + y1r
            X2 = rx1 + x2r
            Y2 = ry1 + y2r
            confb = float(b.conf)

            area_ratio = max(1.0, (X2 - X1) * (Y2 - Y1)) / frame_area
            area_pen = (
                1.0
                if (self.area_ratio_min <= area_ratio <= self.area_ratio_max)
                else 0.5
            )

            orange = self._orange_ratio(frame, (X1, Y1, X2, Y2))
            circ = self._circularity_score(frame, (X1, Y1, X2, Y2))

            # bonus prossimità giocatori (se disponibili)
            prox = self._player_proximity_score((X1, Y1, X2, Y2), players or [], wF, hF)
            iou_prev = self._bbox_iou((X1, Y1, X2, Y2), self.last_bbox)

            if self.select_by_conf:
                fused = self._conf_sigmoid(confb)
            else:
                fused = (
                    self.w_conf * self._conf_sigmoid(confb)
                    + self.w_orange * orange
                    + self.w_shape * circ
                    + self.w_dist * 0.0  # in ROI non usiamo dist_score
                    + self.w_player_prox * prox
                    + self.w_iou * iou_prev
                )
            fused *= area_pen

            if fused > best_score:
                best_score = fused
                best_item = {
                    "bbox": [X1, Y1, X2, Y2],
                    "conf": confb,
                    "score": float(fused),
                    "orange": float(orange),
                    "shape": float(circ),
                    "maha": None,
                    "src": "ROI",
                    "iou_prev": float(iou_prev),
                }
        return best_item

    def _tile_slices(self, h, w, grid=None, overlap=None):
        grid = self.tile_grid if grid is None else grid
        overlap = self.tile_overlap if overlap is None else overlap
        cols, rows = grid
        tiles = []
        # dimensione base senza overlap
        base_w = w / cols
        base_h = h / rows
        # estensione overlap in px
        ow = base_w * overlap
        oh = base_h * overlap
        for r in range(rows):
            for c in range(cols):
                x1 = int(max(0, c * base_w - (ow if c > 0 else 0)))
                y1 = int(max(0, r * base_h - (oh if r > 0 else 0)))
                x2 = int(min(w, (c + 1) * base_w + (ow if c < cols - 1 else 0)))
                y2 = int(min(h, (r + 1) * base_h + (oh if r < rows - 1 else 0)))
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    tiles.append((x1, y1, x2, y2))
        return tiles

    def _tile_redetect(
        self,
        frame,
        grid=None,
        overlap=None,
        imgsz=None,
        conf=None,
        max_det=None,
        players=None,
    ):
        hF, wF = frame.shape[:2]
        tiles = self._tile_slices(hF, wF, grid=grid, overlap=overlap)
        if not tiles:
            return None

        # 1) prepara batch
        imgs, meta = [], []
        for x1t, y1t, x2t, y2t in tiles:
            roi = frame[y1t:y2t, x1t:x2t]
            if roi.size == 0:
                continue
            imgs.append(roi)
            meta.append((x1t, y1t, x2t, y2t))
        if not imgs:
            return None

        # 2) unica inferenza batched
        preds = self.model.predict(
            imgs,
            conf=(self.tile_conf if conf is None else conf),
            iou=self.iou_nms,
            imgsz=(self.tile_imgsz if imgsz is None else imgsz),
            classes=[self.ball_class_id],
            augment=self.use_tta,
            agnostic_nms=self.agnostic_nms,
            max_det=(self.tile_max_det if max_det is None else max_det),
            verbose=False,
        )

        # 3) setup gating KF
        z_pred, S = (None, None)
        S_inv = None
        if self.tracker.kf.initialized:
            z_pred, S = self.tracker.kf.peek(dt=1.0)
            if S is not None:
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S_inv = None

        def mahalanobis2(cx, cy):
            if z_pred is None or S_inv is None:
                return None
            d = np.array([cx - z_pred[0], cy - z_pred[1]], dtype=np.float32).reshape(
                2, 1
            )
            return float((d.T @ S_inv @ d).item())

        pcx = pcy = None
        r2 = None
        if self.last_bbox is not None:
            px1, py1, px2, py2 = self.last_bbox
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            # velocità dal KF
            speed = 0.0
            if self.tracker.kf.initialized:
                vx = float(self.tracker.kf.x[2, 0])
                vy = float(self.tracker.kf.x[3, 0])
                speed = (vx * vx + vy * vy) ** 0.5
            diag = (wF * wF + hF * hF) ** 0.5
            v_norm = speed / max(1.0, diag)
            r = max(self.r_min, min(self.r_k * v_norm + self.r_min, self.r_max))
            r2 = (r * max(wF, hF)) ** 2

        best_item, best_score = None, -1e9
        frame_area = float(hF * wF)

        # 4) rimappatura + scoring
        for p, (x1t, y1t, x2t, y2t) in zip(preds, meta):
            if len(p.boxes) == 0:
                continue
            for b in p.boxes:
                x1r, y1r, x2r, y2r = b.xyxy.squeeze().tolist()
                X1, Y1, X2, Y2 = x1t + x1r, y1t + y1r, x1t + x2r, y1t + y2r
                confb = float(b.conf)
                cx, cy = (X1 + X2) / 2.0, (Y1 + Y2) / 2.0

                area_ratio = max(1.0, (X2 - X1) * (Y2 - Y1)) / frame_area
                area_pen = (
                    1.0
                    if (self.area_ratio_min <= area_ratio <= self.area_ratio_max)
                    else 0.5
                )

                d2 = mahalanobis2(cx, cy)
                maha_pen = math.sqrt(d2) if d2 is not None and d2 >= 0.0 else 0.0
                if d2 is None and pcx is not None and r2 and r2 > 0:
                    dd2 = (cx - pcx) ** 2 + (cy - pcy) ** 2
                    dist_score = max(0.0, 1.0 - (dd2 / r2))
                else:
                    dist_score = 0.0

                orange = self._orange_ratio(frame, (X1, Y1, X2, Y2))
                circ = self._circularity_score(frame, (X1, Y1, X2, Y2))
                iou_prev = self._bbox_iou((X1, Y1, X2, Y2), self.last_bbox)

                prox = self._player_proximity_score(
                    (X1, Y1, X2, Y2), players or [], wF, hF
                )

                if self.select_by_conf:
                    fused = self._conf_sigmoid(confb)
                else:
                    fused = (
                        self.w_conf * self._conf_sigmoid(confb)
                        + self.w_orange * orange
                        + self.w_shape * circ
                        + self.w_dist * dist_score
                        - self.w_maha * maha_pen
                        + self.w_player_prox * prox
                        + self.w_iou * iou_prev
                    )
                fused *= area_pen

                if fused > best_score:
                    best_score = fused
                    best_item = {
                        "bbox": [X1, Y1, X2, Y2],
                        "conf": confb,
                        "score": float(fused),
                        "orange": float(orange),
                        "shape": float(circ),
                        "maha": float(d2) if d2 is not None else None,
                        "src": "TILE",
                        "iou_prev": float(iou_prev),
                    }

        return best_item

    def _circularity_score(self, frame, bbox):
        import cv2

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu per isolare regione principale
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        perim = float(cv2.arcLength(c, True))
        if perim <= 1e-3:
            return 0.0
        circ = 4.0 * math.pi * area / (perim * perim)
        return float(max(0.0, min(1.0, circ)))

    def _bbox_iou(self, a, b):
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
        union = a_area + b_area - inter
        return float(inter / union) if union > 0 else 0.0

    def detect_frames(self, frames, frame_indices=None, players_by_frame=None):
        detections = []
        for i, frame in enumerate(frames):
            # 1) YOLO sul frame intero (molto permissivo grazie a NMS "lasco").
            #    Applica parametri adattivi se abilitati.
            eff = self._effective_params()
            ball_detections = self.model.predict(
                frame,
                conf=eff["min_conf"],
                iou=self.iou_nms,
                imgsz=eff["imgsz"],
                classes=[self.ball_class_id],
                augment=self.use_tta,
                agnostic_nms=self.agnostic_nms,
                max_det=self.max_det,
                verbose=False,
            )
            # 1b) opzionale: rileva i giocatori per prior di prossimità
            players_list = []
            if self.enable_player_proximity:
                # Se fornito dall'esterno (es. PlayerTracker), usa quello
                if players_by_frame is not None:
                    global_idx = frame_indices[i] if frame_indices is not None else i
                    frame_tracks = players_by_frame.get(global_idx, {})
                    for _, info in frame_tracks.items():
                        bboxp = info.get("bbox")
                        if bboxp:
                            players_list.append(list(bboxp))
                # Altrimenti prova a rilevare dal modello corrente, se supporta classi giocatore
                elif self.player_class_ids:
                    p = self.model.predict(
                        frame,
                        conf=self.player_min_conf,
                        iou=self.iou_nms,
                        imgsz=max(self.imgsz, eff["imgsz"]),
                        classes=sorted(list(self.player_class_ids)),
                        augment=False,
                        agnostic_nms=self.agnostic_nms,
                        max_det=self.player_max_det,
                        verbose=False,
                    )
                    if len(p) > 0 and hasattr(p[0], "boxes"):
                        for bb in p[0].boxes:
                            x1p, y1p, x2p, y2p = bb.xyxy.squeeze().tolist()
                            players_list.append([x1p, y1p, x2p, y2p])
            if self.debug_ball:
                nb = int(len(ball_detections[0].boxes))
                top = [float(b.conf) for b in ball_detections[0].boxes[:3]]
                print(
                    f"[Ball YOLO] boxes={nb} top_confs={top} eff={{'conf': {eff['min_conf']:.3f}, 'imgsz': {eff['imgsz']}, 'chi2': {eff['chi2']:.2f}}} players={len(players_list)}"
                )

            chosen = None
            if len(ball_detections[0].boxes) > 0:
                boxes = ball_detections[0].boxes
                h, w = frame.shape[:2]
                frame_area = float(h * w)

                # 2) Predizione KF per gating ellittico (Mahalanobis)
                z_pred, S = (None, None)
                if self.tracker.kf.initialized:
                    z_pred, S = self.tracker.kf.peek(dt=1.0)
                S_inv = None
                if S is not None:
                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        S_inv = None

                # 3) Raggio di ricerca adattivo (in px) attorno all'ultima posizione
                diag = float((w * w + h * h) ** 0.5)
                # speed in px/frame dal KF
                speed = 0.0
                if self.tracker.kf.initialized:
                    vx = float(self.tracker.kf.x[2, 0])
                    vy = float(self.tracker.kf.x[3, 0])
                    speed = math.hypot(vx, vy)
                v_norm = speed / max(1.0, diag)
                radius_frac = max(
                    self.r_min, min(self.r_k * v_norm + self.r_min, self.r_max)
                )
                growth = 1.0 + 0.5 * float(getattr(self.tracker, "misses", 0))
                r2 = (radius_frac * max(w, h) * growth) ** 2

                # centro dell'ultima bbox per la componente distanza euclidea di fallback
                pcx = pcy = None
                if self.last_bbox is not None:
                    lx1, ly1, lx2, ly2 = self.last_bbox
                    pcx, pcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0

                def mahalanobis2(cx, cy):
                    if z_pred is None or S_inv is None:
                        return None
                    d = np.array(
                        [cx - z_pred[0], cy - z_pred[1]], dtype=np.float32
                    ).reshape(2, 1)
                    return float((d.T @ S_inv @ d).item())

                def conf_sigmoid(c):
                    return 1.0 / (1.0 + math.exp(-(float(c) - 0.2) / 0.1))

                best_item = None
                best_score = -1e9

                # 4) Gating Mahalanobis (se disponibile) + Gating Euclideo robusto
                candidate_list = list(boxes)
                if S_inv is not None:
                    gated = []
                    for b in candidate_list:
                        x1, y1, x2, y2 = b.xyxy.squeeze().tolist()
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        d2 = mahalanobis2(cx, cy)
                        if d2 is not None and d2 <= eff["chi2"]:
                            gated.append(b)
                    if gated:
                        candidate_list = gated

                # Gating euclideo hard-limit per evitare salti estremi tra frame
                # Usa centro previsto dal KF se disponibile, altrimenti l'ultimo centro valido.
                misses = int(getattr(self.tracker, "misses", 0))
                eff_radius_frac = min(0.25, float(self.max_search_radius) * (1.0 + 0.25 * misses))
                radius_px = eff_radius_frac * max(w, h)
                if (z_pred is not None or self.last_bbox is not None) and radius_px > 1.0:
                    px, py = None, None
                    if z_pred is not None:
                        px, py = float(z_pred[0]), float(z_pred[1])
                    elif self.last_bbox is not None:
                        lx1, ly1, lx2, ly2 = self.last_bbox
                        px, py = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
                    if px is not None and py is not None:
                        gated = []
                        r2_euc = radius_px * radius_px
                        for b in candidate_list:
                            x1, y1, x2, y2 = b.xyxy.squeeze().tolist()
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            dd2 = (cx - px) * (cx - px) + (cy - py) * (cy - py)
                            if dd2 <= r2_euc:
                                gated.append(b)
                        # Applica il gating solo se sopravvive almeno un candidato
                        if gated:
                            candidate_list = gated

                # 5) Scoring fuso
                for b in candidate_list:
                    x1, y1, x2, y2 = b.xyxy.squeeze().tolist()
                    conf = float(b.conf)
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    area_ratio = max(1.0, (x2 - x1) * (y2 - y1)) / frame_area
                    area_pen = (
                        1.0
                        if (self.area_ratio_min <= area_ratio <= self.area_ratio_max)
                        else 0.5
                    )

                    d2 = mahalanobis2(cx, cy)
                    maha_pen = math.sqrt(d2) if d2 is not None and d2 >= 0.0 else 0.0

                    # fallback distanza euclidea normalizzata quando Mahalanobis non è disponibile
                    if d2 is None and pcx is not None and r2 and r2 > 0:
                        dd2 = (cx - pcx) ** 2 + (cy - pcy) ** 2
                        dist_score = max(0.0, 1.0 - (dd2 / r2))
                    else:
                        dist_score = 0.0

                    orange = self._orange_ratio(frame, (x1, y1, x2, y2))
                    circ = self._circularity_score(frame, (x1, y1, x2, y2))
                    prox = self._player_proximity_score(
                        (x1, y1, x2, y2), players_list, w, h
                    )
                    iou_prev = self._bbox_iou((x1, y1, x2, y2), self.last_bbox)

                    if self.select_by_conf:
                        fused = conf_sigmoid(conf)
                    else:
                        fused = (
                            self.w_conf * conf_sigmoid(conf)
                            + self.w_orange * orange
                            + self.w_shape * circ
                            + self.w_dist * dist_score
                            - self.w_maha * maha_pen
                            + self.w_player_prox * prox
                            + self.w_iou * iou_prev
                        )
                    fused *= area_pen

                    if fused > best_score:
                        best_score = fused
                        best_item = {
                            "bbox": [x1, y1, x2, y2],
                            "conf": conf,
                            "score": float(fused),
                            "orange": float(orange),
                            "shape": float(circ),
                            "maha": float(d2) if d2 is not None else None,
                            "src": "FULL",
                            "iou_prev": float(iou_prev),
                        }

                chosen = best_item

            # Ulteriore controllo di coerenza movimento sul candidato scelto
            if chosen is not None:
                misses = int(getattr(self.tracker, "misses", 0))
                eff_radius_frac = min(0.25, float(self.max_search_radius) * (1.0 + 0.25 * misses))
                radius_px = eff_radius_frac * max(w, h)
                px, py = None, None
                if self.tracker.kf.initialized:
                    z_pred_chk, _ = self.tracker.kf.peek(dt=1.0)
                    if z_pred_chk is not None:
                        px, py = float(z_pred_chk[0]), float(z_pred_chk[1])
                if px is None and self.last_bbox is not None:
                    lx1, ly1, lx2, ly2 = self.last_bbox
                    px, py = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
                if px is not None and py is not None and radius_px > 1.0:
                    x1c, y1c, x2c, y2c = chosen["bbox"]
                    cx, cy = (x1c + x2c) * 0.5, (y1c + y2c) * 0.5
                    dd2 = (cx - px) * (cx - px) + (cy - py) * (cy - py)
                    if dd2 > (radius_px * radius_px):
                        # troppo lontano => forza fallback successivi
                        chosen = None

            if chosen is not None:
                x1, y1, x2, y2 = chosen["bbox"]
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                ps = self.tm_patch_size
                px1, py1 = int(max(0, cx - ps / 2)), int(max(0, cy - ps / 2))
                px2, py2 = int(min(w, px1 + ps)), int(min(h, py1 + ps))
                patch = frame[py1:py2, px1:px2]
                if patch.size > 0 and patch.shape[:2] == (ps, ps):
                    self.prev_patch = patch.copy()

            # 6) Fallback ROI se nulla scelto e abbiamo una bbox precedente
            if chosen is None and self.last_bbox is not None:
                # conf come fattore relativo a self.min_conf (vedi _roi_redetect)
                roi_factor = eff["roi_conf_factor"] * (
                    0.8 if getattr(self.tracker, "misses", 0) >= 3 else 1.0
                )
                zoom_item = self._roi_redetect(
                    frame,
                    self.last_bbox,
                    conf=roi_factor,
                    scale=eff["roi_scale"],
                    roi_limit=self.roi_limit,
                    players=players_list,
                )
                if zoom_item is not None:
                    chosen = zoom_item
            # 7) Fallback TILES se ancora nullo (o se non abbiamo storia per ROI)
            if chosen is None:
                tile_item = self._tile_redetect(
                    frame,
                    grid=eff["tile_grid"],
                    overlap=eff["tile_overlap"],
                    imgsz=eff["tile_imgsz"],
                    conf=eff["tile_conf"],
                    players=players_list,
                )
                if tile_item is not None:
                    chosen = tile_item

            if chosen is None:
                # centro previsto dal KF o dall’ultima bbox
                pred_center = None
                if self.tracker.kf.initialized:
                    z_pred, _ = self.tracker.kf.peek(dt=1.0)
                    if z_pred is not None:
                        pred_center = (float(z_pred[0]), float(z_pred[1]))
                if pred_center is None and self.last_bbox is not None:
                    lx1, ly1, lx2, ly2 = self.last_bbox
                    pred_center = ((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0)
                if pred_center is not None:
                    search_px = int(self.r_min * max(w, h))
                    fbbox, score = self._tm_fallback(frame, pred_center, search_px)
                    if fbbox is not None:
                        chosen = {
                            "bbox": fbbox,
                            "conf": 0.05,
                            "score": score,
                            "orange": 0.0,
                            "shape": 0.0,
                            "maha": None,
                            "src": "TM",
                            "iou_prev": self._bbox_iou(fbbox, self.last_bbox),
                        }
                        # aggiorna patch
                        cx, cy = (fbbox[0] + fbbox[2]) / 2.0, (
                            fbbox[1] + fbbox[3]
                        ) / 2.0
                        ps = self.tm_patch_size
                        px1, py1 = int(max(0, cx - ps / 2)), int(max(0, cy - ps / 2))
                        px2, py2 = int(min(w, px1 + ps)), int(min(h, py1 + ps))
                        patch = frame[py1:py2, px1:px2]
                        if (
                            patch.size > 0
                            and patch.shape[0] == ps
                            and patch.shape[1] == ps
                        ):
                            self.prev_patch = patch.copy()

            # 8) se il miglior punteggio è troppo basso, forza fallback
            if chosen is not None and self.min_fused_select > 0.0:
                try:
                    if float(chosen.get("score", 0.0)) < float(self.min_fused_select):
                        chosen = None
                except Exception:
                    pass

            # 9) metrics + append per ogni frame
            try:
                self.metrics.append(
                    {
                        "src": chosen.get("src") if chosen else None,
                        "conf": float(chosen.get("conf", 0.0)) if chosen else None,
                        "score": float(chosen.get("score", 0.0)) if chosen else None,
                        "orange": float(chosen.get("orange", 0.0)) if chosen else None,
                        "shape": float(chosen.get("shape", 0.0)) if chosen else None,
                        "maha": (
                            float(chosen.get("maha"))
                            if chosen and chosen.get("maha") is not None
                            else None
                        ),
                        "misses": int(getattr(self.tracker, "misses", 0)),
                        "eff_min_conf": float(eff["min_conf"]),
                        "eff_imgsz": int(eff["imgsz"]),
                        "eff_chi2": float(eff["chi2"]),
                        "nb_boxes_full": int(len(ball_detections[0].boxes)),
                        "nb_players": int(len(players_list)),
                        "iou_prev": (
                            float(chosen.get("iou_prev", 0.0)) if chosen else None
                        ),
                    }
                )
            except Exception:
                pass
            detections.append(chosen)

        # fuori dal for: ritorna tutte le detections
        return detections

    def export_metrics_csv(self, path, frame_indices=None):
        import csv, os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        headers = [
            "frame",
            "src",
            "conf",
            "score",
            "orange",
            "shape",
            "maha",
            "misses",
            "eff_min_conf",
            "eff_imgsz",
            "eff_chi2",
            "nb_boxes_full",
            "nb_players",
        ]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for i, m in enumerate(self.metrics):
                frame_id = (
                    frame_indices[i]
                    if frame_indices is not None and i < len(frame_indices)
                    else i
                )
                w.writerow(
                    [
                        frame_id,
                        m.get("src"),
                        m.get("conf"),
                        m.get("score"),
                        m.get("orange"),
                        m.get("shape"),
                        m.get("maha"),
                        m.get("misses"),
                        m.get("eff_min_conf"),
                        m.get("eff_imgsz"),
                        m.get("eff_chi2"),
                        m.get("nb_boxes_full"),
                        m.get("nb_players"),
                    ]
                )

    def export_metrics_json(self, path, frame_indices=None):
        import json, os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = []
        for i, m in enumerate(self.metrics):
            frame_id = (
                frame_indices[i]
                if frame_indices is not None and i < len(frame_indices)
                else i
            )
            mm = dict(m)
            mm["frame"] = frame_id
            out.append(mm)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def update_tracks(self, detections):
        for frame_num, det in enumerate(detections):
            if not det:
                # prova a predire senza misura
                tracked_bbox = self.tracker.process_detections(None, 0.0)
                if tracked_bbox:
                    self.tracks[frame_num] = {
                        1: {
                            "bbox": tracked_bbox,
                            "score": 0.0,
                            "class_id": self.ball_class_id,
                            "state": "PRED",
                            "tracker_misses": self.tracker.misses,
                        }
                    }
                    self.last_bbox = tracked_bbox
                else:
                    self.tracks[frame_num] = {}
                continue

            bbox = det["bbox"]
            conf = float(det.get("conf", 0.0))

            # Soppressione dimensionale per evitare salti di scala improbabili
            if self.last_bbox is not None:
                prev_w = self.last_bbox[2] - self.last_bbox[0]
                prev_h = self.last_bbox[3] - self.last_bbox[1]
                prev_area = max(1.0, prev_w * prev_h)
                w = max(1.0, bbox[2] - bbox[0])
                h = max(1.0, bbox[3] - bbox[1])
                area = w * h
                if area > 3.0 * prev_area or area < 0.2 * prev_area:
                    tracked_bbox = self.tracker.process_detections(None, conf)
                else:
                    tracked_bbox = self.tracker.process_detections(bbox, conf)
            else:
                tracked_bbox = self.tracker.process_detections(bbox, conf)

            if tracked_bbox:
                info = {
                    "bbox": tracked_bbox,
                    "score": conf,
                    "class_id": self.ball_class_id,
                    "state": "ACTIVE",
                    "tracker_misses": self.tracker.misses,
                }
                # Propaga metriche se presenti
                for k in ("maha", "orange", "shape"):
                    if det.get(k) is not None:
                        info[k] = det[k]
                if det.get("src") is not None:
                    info["src"] = det["src"]
                self.tracks[frame_num] = {1: info}
                self.last_bbox = tracked_bbox
            else:
                self.tracks[frame_num] = {}

    # --- in BallTracker.track_frames ---
    def track_frames(
        self, frames, read_from_stub=False, stub_path=None, frame_indices=None, players_by_frame=None
    ):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Caricamento tracce della palla da stub: {stub_path}")
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        print(
            "Esecuzione del tracking della palla (nessuno stub trovato o richiesto)..."
        )
        detections = self.detect_frames(frames, frame_indices=frame_indices, players_by_frame=players_by_frame)  # mantiene l’ordine!
        # salva tracce usando gli indici globali se forniti
        self.tracks = {}
        for i, det in enumerate(detections):
            global_idx = frame_indices[i] if frame_indices is not None else i

            if det:
                bbox = det["bbox"]
                conf = float(det.get("conf", 0.0))

                # Soppressione dimensionale per evitare salti di scala improbabili
                if self.last_bbox is not None:
                    prev_w = self.last_bbox[2] - self.last_bbox[0]
                    prev_h = self.last_bbox[3] - self.last_bbox[1]
                    prev_area = max(1.0, prev_w * prev_h)
                    w = max(1.0, bbox[2] - bbox[0])
                    h = max(1.0, bbox[3] - bbox[1])
                    area = w * h
                    if area > 3.0 * prev_area or area < 0.2 * prev_area:
                        tracked_bbox = self.tracker.process_detections(None, conf)
                    else:
                        tracked_bbox = self.tracker.process_detections(bbox, conf)
                else:
                    tracked_bbox = self.tracker.process_detections(bbox, conf)

                if tracked_bbox:
                    info = {
                        "bbox": tracked_bbox,
                        "score": conf,
                        "class_id": self.ball_class_id,
                        "state": "ACTIVE",
                        "tracker_misses": self.tracker.misses,
                    }
                    # Propaga metriche se presenti
                    for k in ("maha", "orange", "shape"):
                        if det.get(k) is not None:
                            info[k] = det[k]
                    if det.get("src") is not None:
                        info["src"] = det["src"]
                    self.tracks[global_idx] = {1: info}
                    self.last_bbox = tracked_bbox
                else:
                    self.tracks[global_idx] = {}
            else:
                tracked_bbox = self.tracker.process_detections(None, 0.0)
                if tracked_bbox:
                    self.tracks[global_idx] = {
                        1: {
                            "bbox": tracked_bbox,
                            "score": 0.0,
                            "class_id": self.ball_class_id,
                            "state": "PRED",
                            "tracker_misses": self.tracker.misses,
                        }
                    }
                    self.last_bbox = tracked_bbox
                else:
                    self.tracks[global_idx] = {}

        if stub_path:
            print(f"Salvataggio tracce della palla in stub: {stub_path}")
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(self.tracks, f)
        return self.tracks
