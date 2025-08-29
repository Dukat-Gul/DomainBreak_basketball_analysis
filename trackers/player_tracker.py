# FILE: trackers/player_tracker.py

import pickle
import os
from ultralytics import YOLO


class PlayerTracker:
    def __init__(self, model, imgsz=None):
        self.model = model
        self.imgsz = imgsz

    def detect_frames(self, frames):
        detections = []
        for frame in frames:
            # MODIFICA: Rimuovere 'classes=[0]' per rilevare tutte le classi
            if self.imgsz is not None:
                player_detections = self.model.track(
                    frame, persist=True, conf=0.15, imgsz=self.imgsz
                )[0]
            else:
                player_detections = self.model.track(
                    frame, persist=True, conf=0.15
                )[0]
            detections.append(player_detections)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Caricamento tracce dei giocatori da stub: {stub_path}")
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        print(
            "Esecuzione del tracking dei giocatori (nessuno stub trovato o richiesto)..."
        )
        detections = self.detect_frames(frames)

        tracks = {}
        for frame_num, detection in enumerate(detections):
            tracks[frame_num] = {}
            if detection.boxes.id is not None:
                for box in detection.boxes:
                    track_id = int(box.id.item())
                    bbox = box.xyxy.squeeze().tolist()
                    score = float(box.conf.item())
                    # Aggiungiamo anche la classe rilevata per un filtraggio futuro
                    class_id = int(box.cls.item())
                    tracks[frame_num][track_id] = {
                        "bbox": bbox,
                        "score": score,
                        "class_id": class_id,
                    }

        if stub_path:
            print(f"Salvataggio tracce dei giocatori in stub: {stub_path}")
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
