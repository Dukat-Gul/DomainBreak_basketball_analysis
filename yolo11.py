# filepath: c:\Users\23ema\OneDrive\Documenti\basketball_analysis\basketball_analysis\yolo11.py
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11x.pt")  # Usa il modello più grande per miglior precisione

# Run tracking with the YOLO11n model on the video
results = model.track(
    "input_videos\MeloCrazy3Shot.mp4",
    classes=[32],
    save=True,
    show=True,
    conf=0.3,
    iou=0.5,
    max_det=1,
    device="cuda",  # Usa la GPU
    line_width=2,
    save_json=True,
    stream=True,
)

for result in results:
    # Processa ogni frame
    print(f"Frame elaborato: {result}")