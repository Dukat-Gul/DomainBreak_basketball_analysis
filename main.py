import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Optional
from utils import read_video, save_video, save_stub
from trackers import PlayerTracker, BallTracker

from ultralytics import YOLO

# ... (altri import come prima)
from court_keypoint_detector import CourtKeypointDetector
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector
from shot_detector import ShotDetector
from shot_classifier import ShotClassifier
from tactical_view import TacticalViewConverter
from shot_visualizer import ShotVisualizer
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
)
from configs import (
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
)


# Add this function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def resolve_device(user_device: Optional[str] = None) -> str:
    """Resolve device string from user input, defaulting to best available.

    - 'auto' or None -> 'cuda' if available else 'cpu'
    - otherwise returns the provided value (e.g., 'cuda', 'cuda:0', 'cpu')
    """
    if user_device and user_device.lower() != "auto":
        return user_device
    try:
        import torch  # local import to avoid hard dependency if not installed yet

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main(input_video_path, output_video_path, read_stubs=False):
    video_frames = read_video(input_video_path)
    # Determina il device per i modelli YOLO
    device = resolve_device(getattr(args, "device", "auto"))
    print(f"[Device] Inference device: {device}")

    # --- Analisi dei Giocatori e del Campo (invariata) ---
    stubs_dir = "stubs"
    os.makedirs(stubs_dir, exist_ok=True)
    player_tracks_stub = os.path.join(stubs_dir, "player_track_stubs.pkl")
    court_keypoints_stub = os.path.join(stubs_dir, "court_key_points_stub.pkl")
    # ... (il resto della definizione dei percorsi stub è invariato)
    player_assignment_stub = os.path.join(stubs_dir, "player_assignment_stub.pkl")
    ball_acquisition_stub = os.path.join(stubs_dir, "ball_acquisition.pkl")

    # Load model object first, then create tracker
    player_model = YOLO(PLAYER_DETECTOR_PATH)
    try:
        player_model.to(device)
    except Exception:
        pass
    player_tracker = PlayerTracker(player_model)
    player_tracks = player_tracker.get_object_tracks(
        video_frames, read_from_stub=read_stubs, stub_path=player_tracks_stub
    )

    court_keypoint_detector = CourtKeypointDetector(
        COURT_KEYPOINT_DETECTOR_PATH, device=device
    )
    court_keypoints = court_keypoint_detector.get_court_keypoints(
        video_frames, read_from_stub=read_stubs, stub_path=court_keypoints_stub
    )

    team_assigner = TeamAssigner()
    player_assignments = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stub=read_stubs,
        stub_path=player_assignment_stub,
    )

    # --- Rilevamento della Palla (con il nuovo metodo pulito) ---
    ball_model = YOLO(BALL_DETECTOR_PATH)
    try:
        ball_model.to(device)
    except Exception:
        pass
    ball_tracker = BallTracker(
        ball_model,
        imgsz=args.ball_pred_imgsz,
        min_conf=args.ball_min_conf,
        use_tta=args.ball_use_tta,
        select_by_conf=args.ball_select_by_conf,
        max_search_radius=args.ball_search_radius,
        kalman_max_misses=args.ball_kf_max_misses,
    )
    # Soglia minima del punteggio fuso per accettare un candidato
    ball_tracker.min_fused_select = float(args.ball_min_fused_select)
    ball_tracker.w_iou = args.ball_w_iou
    # MODIFICA: Chiamiamo il nuovo metodo 'track_frames' se non leggiamo da stub
    if read_stubs:
        with open(os.path.join(stubs_dir, "ball_track_stubs.pkl"), "rb") as f:
            ball_tracks = pd.read_pickle(f)
    else:
        frame_indices = list(range(len(video_frames)))
        ball_tracks = ball_tracker.track_frames(
            video_frames,
            frame_indices=frame_indices,
            players_by_frame=player_tracks,
        )
        save_stub(os.path.join(stubs_dir, "ball_track_stubs.pkl"), ball_tracks)

    # --- Rilevamento Possesso Palla (invariato) ---
    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(
        player_tracks, ball_tracks
    )
    if not read_stubs:
        save_stub(ball_acquisition_stub, ball_acquisition)

    # --- Inizializzazione (invariata) ---
    # ... (il codice da qui è identico a prima fino alla fine)
    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()
    court_drawer = CourtKeypointDrawer()
    shot_detector = ShotDetector(basket_area=None)
    shot_classifier = ShotClassifier(
        court_dimensions=None
    )  # Per ora non passiamo le dimensioni reali
    shot_visualizer = ShotVisualizer()
    tactical_converter = TacticalViewConverter()
    shot_outcome_info = {"text": "", "display_frames": 0}
    output_frames = []

    # NUOVO: Dizionario per raccogliere i dati di analisi
    analysis_data = {}

    print("Inizio elaborazione e disegno frame per frame...")
    for frame_num, frame in enumerate(video_frames):
        player_tracks_for_frame = player_tracks[frame_num]
        ball_tracks_for_frame = ball_tracks[frame_num]
        court_keypoints_for_frame = court_keypoints[frame_num]
        player_assignments_for_frame = player_assignments[frame_num]
        player_with_ball_id = ball_acquisition[frame_num]

        # NUOVO: Raccogliamo i dati per il JSON
        frame_analysis = {
            "frame_number": frame_num,
            "player_with_ball": player_with_ball_id,
            "players": [],
            "ball": [],
            "events": [],
        }

        # Aggiungi dati giocatori
        for player_id, track_info in player_tracks_for_frame.items():
            team_id = player_assignments_for_frame.get(player_id)
            player_data = {
                "id": player_id,
                "team_id": team_id if team_id is not None else -1,
                "position_2d": track_info.get("bbox", []),
            }
            frame_analysis["players"].append(player_data)

        # Aggiungi dati palla
        for ball_id, track_info in ball_tracks_for_frame.items():
            ball_data = {"id": ball_id, "position_2d": track_info.get("bbox", [])}
            frame_analysis["ball"].append(ball_data)

        basket_2d_coords = tactical_converter.transform_3d_to_2d(
            tactical_converter.basket_3d_coordinates, court_keypoints_for_frame
        )
        if basket_2d_coords is not None:
            shot_detector.basket_area = basket_2d_coords[0]
        shot_event = shot_detector.detect(
            ball_tracks_for_frame,
            player_tracks_for_frame,
            player_with_ball_id,
            frame_num,
        )

        # NUOVO: Aggiungi eventi al JSON
        if shot_event:
            frame_analysis["events"].append(shot_event)

        if shot_event and shot_event.get("shot_event") == "shot_ended":
            # Classifichiamo il tiro solo se è andato a segno
            shot_type = ""
            if shot_event.get("successful") and shot_event.get("shot_position"):
                classification = shot_classifier.classify(
                    shot_event, shot_event["shot_position"]
                )
                shot_type = f" ({classification['shot_type']})"

            # Aggiorniamo il testo da visualizzare
            outcome = "Bucket!" if shot_event.get("successful") else "Miss!"
            shot_outcome_info["text"] = f"{outcome}{shot_type}"
            shot_outcome_info["display_frames"] = (
                60  # Mostra il testo per 2 secondi (a 30fps)
            )

        analysis_data[frame_num] = frame_analysis

        annotated_frame = frame.copy()
        annotated_frame = court_drawer.draw_frame(
            annotated_frame, court_keypoints_for_frame
        )
        annotated_frame = player_drawer.draw_frame(
            annotated_frame,
            player_tracks_for_frame,
            player_assignments_for_frame,
            player_with_ball_id,
        )
        annotated_frame = ball_drawer.draw_frame(annotated_frame, ball_tracks_for_frame)
        outcome_text_to_draw = None
        if shot_outcome_info["display_frames"] > 0:
            outcome_text_to_draw = shot_outcome_info["text"]
            shot_outcome_info["display_frames"] -= 1
        annotated_frame = shot_visualizer.draw(
            annotated_frame, shot_detector, outcome_text_to_draw
        )
        output_frames.append(annotated_frame)

    print("Elaborazione completata. Salvataggio del video e dei dati di analisi...")

    # Salvataggio video
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    save_video(output_frames, output_video_path)
    print(f"✅ Video finale salvato in '{output_video_path}'")

    output_json_path = os.path.splitext(output_video_path)[0] + ".json"
    with open(output_json_path, "w") as f:
        # Convert NumPy types before serialization
        serializable_data = convert_numpy_types(analysis_data)
        json.dump(serializable_data, f, indent=4)
    print(f"✅ Dati di analisi salvati in '{output_json_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esegue l'analisi video di una partita di basket."
    )
    parser.add_argument(
        "--input_video", type=str, required=True, help="Percorso del video di input."
    )
    parser.add_argument(
        "--output_video",
        type=str,
        required=True,
        help="Percorso dove salvare il video analizzato.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device per l'inferenza YOLO: 'auto', 'cpu', 'cuda', 'cuda:0', ecc.",
    )
    parser.add_argument(
        "--no_stubs",
        action="store_true",
        help="Forza la ri-analisi ignorando i file stub.",
    )
    # Parametri BallTracker per allineare a evaluation.py
    parser.add_argument(
        "--ball_pred_imgsz",
        type=int,
        default=960,
        help="Dimensione di inferenza YOLO per la palla",
    )
    parser.add_argument(
        "--ball_min_conf",
        type=float,
        default=0.05,
        help="Soglia conf minima per la palla",
    )
    parser.add_argument(
        "--ball_use_tta",
        action="store_true",
        help="Abilita test-time augmentation per la palla",
    )
    parser.add_argument(
        "--ball_w_iou",
        type=float,
        default=0.20,
        help="Peso del termine IoU con la bbox precedente nello scoring",
    )
    parser.add_argument(
        "--ball_min_fused_select",
        type=float,
        default=0.0,
        help="Soglia minima del punteggio fuso per accettare un candidato (0=disabled)",
    )
    parser.add_argument(
        "--ball_select_by_conf",
        action="store_true",
        help="Se impostato, seleziona la bbox in base alla sola confidenza (post-gating)",
    )
    parser.add_argument(
        "--ball_search_radius",
        type=float,
        default=0.10,
        help="Raggio di ricerca euclideo massimo per frame (frazione di max(lato))",
    )
    parser.add_argument(
        "--ball_kf_max_misses",
        type=int,
        default=8,
        help="Quanti frame mantenere la traccia senza detection (predizione KF)",
    )
    args = parser.parse_args()
    main(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        read_stubs=not args.no_stubs,
    )
