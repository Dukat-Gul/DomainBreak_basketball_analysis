import pickle
import os
from ultralytics import YOLO

# Corretto il nome della classe importata
from utils.kalman_filter import DetectionsToTracksKalmanFilter


class BallTracker:
    def __init__(self, model):
        self.model = model
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

    def detect_frames(self, frames):
        detections = []
        for frame in frames:
            # Filtra direttamente per la classe 'ball'
            ball_detections = self.model.predict(
                frame, conf=0.15, classes=[self.ball_class_id]
            )
            ball_detections_bbox = []
            if len(ball_detections[0].boxes) > 0:
                highest_conf_detection = max(
                    ball_detections[0].boxes, key=lambda x: x.conf
                )
                ball_detections_bbox.append(highest_conf_detection)
            detections.append(ball_detections_bbox)
        return detections

    def update_tracks(self, detections):
        for frame_num, detection in enumerate(detections):
            if not detection:
                self.tracks[frame_num] = {}
                continue

            bbox = detection[0].xyxy.squeeze().tolist()
            conf = detection[0].conf.squeeze().tolist()
            tracked_bbox = self.tracker.process_detections(bbox, conf)

            if tracked_bbox:
                # Aggiungiamo anche la class_id per la valutazione
                self.tracks[frame_num] = {
                    1: {"bbox": tracked_bbox, "score": conf, "class_id": self.ball_class_id}
                }
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
