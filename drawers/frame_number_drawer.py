import cv2


class FrameNumberDrawer:
    def __init__(self):
        pass

    def draw_frame(self, frame, index: int):
        """Disegna il numero di frame in alto a sinistra su un singolo frame."""
        out = frame.copy()
        cv2.putText(out, str(index), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return out

    def draw(self, frames):
        # Write the frame number on the top left corner of the frame
        output_frames = []
        for i in range(len(frames)):
            output_frames.append(self.draw_frame(frames[i], i))
        return output_frames
