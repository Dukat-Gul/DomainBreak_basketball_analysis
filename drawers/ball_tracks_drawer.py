from .utils import draw_triangle
import cv2
import numpy as np
from collections import deque
from utils import get_center_of_bbox


class BallTracksDrawer:
    """
    Classe responsabile per disegnare il tracciamento della palla sui frame del video.
    """

    def __init__(self, trail_len: int = 25, trail_color=(0, 165, 255), trail_thickness: int = 2):
        self.ball_pointer_color = (
            0,
            255,
            0,
        )  # Colore verde per il puntatore della palla
        # Parametri della scia
        self.trail_len = int(trail_len)
        self.trail_color = tuple(trail_color)
        self.trail_thickness = int(trail_thickness)
        self._trail = deque(maxlen=self.trail_len)

    def draw_frame(self, frame, ball_tracks_for_frame):
        """
        Disegna il puntatore della palla su un SINGOLO frame.

        Args:
            frame (np.array): Il frame su cui disegnare.
            ball_tracks_for_frame (dict): Dizionario con le informazioni di tracciamento della palla.

        Returns:
            np.array: Il frame con il puntatore disegnato.
        """
        output_frame = frame.copy()

        # Disegna il puntatore per la palla (di solito ha ID 1)
        for _, ball_data in ball_tracks_for_frame.items():
            if ball_data.get("bbox"):
                output_frame = draw_triangle(
                    output_frame, ball_data["bbox"], self.ball_pointer_color
                )

        return output_frame

    def draw(self, video_frames, tracks):
        """
        Metodo originale che ora utilizza 'draw_frame' in un ciclo.
        Disegna i puntatori della palla su una lista di frame.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            ball_tracks_for_frame = tracks[frame_num]
            # Aggiorna la scia con il centro della bbox (se presente)
            center = None
            for _, ball_data in ball_tracks_for_frame.items():
                bbox = ball_data.get("bbox")
                if bbox:
                    cx, cy = get_center_of_bbox(bbox)
                    center = (int(cx), int(cy))
                    break
            if center is not None:
                self._trail.append(center)
            else:
                # opzionale: attenua la scia se la palla non è visibile
                if self._trail:
                    self._trail.append(self._trail[-1])

            drawn_frame = self.draw_frame(frame, ball_tracks_for_frame)

            # Disegna la scia (polilinea + piccoli cerchi)
            if len(self._trail) >= 2:
                pts = np.array(list(self._trail), dtype=np.int32)
                cv2.polylines(
                    drawn_frame,
                    [pts],
                    isClosed=False,
                    color=self.trail_color,
                    thickness=self.trail_thickness,
                    lineType=cv2.LINE_AA,
                )
                # Cerchi decrescenti per dare effetto fading semplice
                n = len(pts)
                for i, (x, y) in enumerate(pts):
                    r = max(1, int(4 * (i + 1) / n))
                    cv2.circle(drawn_frame, (x, y), r, self.trail_color, -1, cv2.LINE_AA)

            output_video_frames.append(drawn_frame)

        return output_video_frames
