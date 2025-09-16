from ultralytics import YOLO
import supervision as sv
from typing import List, Optional
import sys
sys.path.append('../')
from utils import read_stub, save_stub


class CourtKeypointDetector:
    """
    The CourtKeypointDetector class uses a YOLO model to detect court keypoints in image frames. 
    It also provides functionality to draw these detected keypoints on the frames.
    """
    def __init__(self, model_path: str, device: Optional[str] = None, conf: float = 0.5, batch_size: int = 20):
        self.model = YOLO(model_path)
        # Sposta il modello sul device richiesto se specificato (es. 'cuda', 'cuda:0', 'cpu')
        if device is not None:
            try:
                self.model.to(device)
            except Exception:
                # In caso di device non disponibile, resta sul default senza interrompere il flusso
                pass
        self.conf = float(conf)
        self.batch_size = int(batch_size)
    
    def get_court_keypoints(self, frames: List, read_from_stub: bool = False, stub_path: Optional[str] = None):
        """
        Detect court keypoints for a batch of frames using the YOLO model. If requested, 
        attempts to read previously detected keypoints from a stub file before running the model.

        Args:
            frames (list of numpy.ndarray): A list of frames (images) on which to detect keypoints.
            read_from_stub (bool, optional): Indicates whether to read keypoints from a stub file 
                instead of running the detection model. Defaults to False.
            stub_path (str, optional): The file path for the stub file. If None, a default path may be used. 
                Defaults to None.

        Returns:
            list: A list of detected keypoints for each input frame.
        """
        court_keypoints = read_stub(read_from_stub, stub_path)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):
                return court_keypoints
        
        court_keypoints = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(
                frames[i : i + self.batch_size], conf=self.conf, verbose=False
            )
            for detection in detections_batch:
                # detection.keypoints potrebbe essere None
                court_keypoints.append(getattr(detection, "keypoints", None))

        save_stub(stub_path, court_keypoints)
        
        return court_keypoints
