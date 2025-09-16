import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Optional
from utils import read_video_with_meta, save_video_with_fps, save_stub
from trackers import PlayerTracker, BallTracker

from ultralytics import YOLO

# ... (altri import come prima)
from court_keypoint_detector import CourtKeypointDetector
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector
from shot_detector import ShotDetector
from shot_classifier import ShotClassifier
from tactical_view_converter.unified_tactical_mapper import UnifiedTacticalMapper
from shot_visualizer import ShotVisualizer
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    FrameNumberDrawer,
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
    video_frames, fps, _ = read_video_with_meta(input_video_path)
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
    player_tracker = PlayerTracker(
        player_model, imgsz=args.player_imgsz, conf=args.player_conf
    )
    player_tracks = player_tracker.get_object_tracks(
        video_frames, read_from_stub=read_stubs, stub_path=player_tracks_stub
    )

    court_keypoint_detector = CourtKeypointDetector(
        COURT_KEYPOINT_DETECTOR_PATH,
        device=device,
        conf=args.court_kp_conf,
        batch_size=args.court_kp_batch,
    )
    court_keypoints = court_keypoint_detector.get_court_keypoints(
        video_frames, read_from_stub=read_stubs, stub_path=court_keypoints_stub
    )
    # Unificato: validazione keypoints per omografie stabili (opt-out via CLI)
    tactical_mapper = UnifiedTacticalMapper()
    if not args.no_kp_validate:
        court_keypoints = (
            tactical_mapper.validate_keypoints(court_keypoints) or court_keypoints
        )

    team_assigner = TeamAssigner(device=device, use_half=args.team_assigner_half)
    player_assignments = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stub=read_stubs,
        stub_path=player_assignment_stub,
        stride=args.team_assigner_stride,
        batch=args.team_assigner_batch,
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
    # ROI/tile params
    ball_tracker.roi_scale = args.ball_roi_scale
    ball_tracker.roi_limit = args.ball_roi_limit
    ball_tracker.roi_conf_factor = args.ball_roi_conf_factor
    ball_tracker.tile_grid = (args.ball_tile_cols, args.ball_tile_rows)
    ball_tracker.tile_overlap = args.ball_tile_overlap
    ball_tracker.tile_imgsz = args.ball_tile_imgsz
    ball_tracker.tile_conf = args.ball_tile_conf
    ball_tracker.tile_max_det = args.ball_tile_max_det
    if args.ball_no_adaptive:
        ball_tracker.adaptive = False
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

    # --- Rilevamento Possesso Palla ---
    ball_acquisition_detector = BallAcquisitionDetector(
        possession_threshold=args.possession_threshold,
        min_frames=args.pos_min_frames,
        containment_threshold=args.pos_containment_threshold,
    )
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(
        player_tracks, ball_tracks
    )
    if not read_stubs:
        save_stub(ball_acquisition_stub, ball_acquisition)

    # --- Pre-computi: posizioni tattiche e pass/intercetti ---
    player_tracks_list = [player_tracks.get(i, {}) for i in range(len(video_frames))]
    tactical_player_positions = tactical_mapper.transform_players_to_tactical_view(
        court_keypoints, player_tracks_list
    )
    try:
        from pass_and_interception_detector import PassAndInterceptionDetector

        pi_detector = PassAndInterceptionDetector()
        passes = pi_detector.detect_passes(ball_acquisition, player_assignments)
        interceptions = pi_detector.detect_interceptions(
            ball_acquisition, player_assignments
        )
    except Exception:
        passes = [-1] * len(video_frames)
        interceptions = [-1] * len(video_frames)

    # --- Inizializzazione (invariata) ---
    # ... (il codice da qui è identico a prima fino alla fine)
    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer(
        trail_len=args.ball_trail_len,
        trail_thickness=args.ball_trail_thickness,
        enable_trail=(not args.no_ball_trail),
    )
    court_drawer = CourtKeypointDrawer()
    frame_id_drawer = FrameNumberDrawer() if args.draw_frame_numbers else None
    shot_detector = ShotDetector(basket_area=None)
    shot_classifier = ShotClassifier(
        court_dimensions=None
    )  # Per ora non passiamo le dimensioni reali
    shot_visualizer = ShotVisualizer(enable_trajectory=(not args.no_shot_trajectory))
    # tactical_mapper già creato
    shot_outcome_info = {"text": "", "display_frames": 0}
    output_frames = []

    # NUOVO: Dizionario per raccogliere i dati di analisi
    analysis_data = {}

    print("Inizio elaborazione e disegno frame per frame...")

    # Helper per accedere sia a dict {frame->...} sia a list [...]
    def get_frame_item(container, index, default=None):
        if isinstance(container, dict):
            return container.get(index, default)
        try:
            return container[index]
        except Exception:
            return default

    for frame_num, frame in enumerate(video_frames):
        player_tracks_for_frame = get_frame_item(player_tracks, frame_num, {})
        ball_tracks_for_frame = get_frame_item(ball_tracks, frame_num, {})
        court_keypoints_for_frame = get_frame_item(court_keypoints, frame_num, None)
        player_assignments_for_frame = get_frame_item(player_assignments, frame_num, {})
        player_with_ball_id = get_frame_item(ball_acquisition, frame_num, -1)

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

        basket_2d_coords = tactical_mapper.transform_3d_to_2d(
            tactical_mapper.basket_3d_coordinates, court_keypoints_for_frame
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
            if shot_event.get("successful"):
                # Prova classificazione su vista tattica (distanza da arco 6.75m)
                start_f = int(shot_event.get("start_frame", frame_num))
                pid = shot_event.get("player_id")
                player_tact = None
                if 0 <= start_f < len(tactical_player_positions):
                    player_tact = tactical_player_positions[start_f].get(pid)
                if player_tact is not None:
                    classification = shot_classifier.classify_tactical(
                        player_tact, tactical_mapper
                    )
                    shot_type = f" ({classification['shot_type']})"
                elif shot_event.get("shot_position"):
                    # Fallback 2D
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

        # Pass/intercetti per frame in JSON
        if passes and 0 <= frame_num < len(passes) and passes[frame_num] in (1, 2):
            frame_analysis["events"].append(
                {"type": "pass", "team": int(passes[frame_num])}
            )
        if (
            interceptions
            and 0 <= frame_num < len(interceptions)
            and interceptions[frame_num] in (1, 2)
        ):
            frame_analysis["events"].append(
                {"type": "interception", "team": int(interceptions[frame_num])}
            )

        # Posizioni tattiche per analisi
        if 0 <= frame_num < len(tactical_player_positions):
            frame_analysis["players_tactical"] = tactical_player_positions[frame_num]

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
        if frame_id_drawer is not None:
            annotated_frame = frame_id_drawer.draw_frame(annotated_frame, frame_num)
        output_frames.append(annotated_frame)

    print("Elaborazione completata. Salvataggio del video e dei dati di analisi...")

    # Overlays post-loop opzionali: speed/distance, pass/intercetti, team control
    if args.draw_speed_distance:
        try:
            from speed_and_distance_calculator import SpeedAndDistanceCalculator
            from drawers import SpeedAndDistanceDrawer

            calc = SpeedAndDistanceCalculator(
                width_in_pixels=tactical_mapper.width_px,
                height_in_pixels=tactical_mapper.height_px,
                width_in_meters=tactical_mapper.width_m,
                height_in_meters=tactical_mapper.height_m,
            )
            distances = calc.calculate_distance(tactical_player_positions)
            speeds = calc.calculate_speed(distances, fps=fps)

            speed_drawer = SpeedAndDistanceDrawer()
            output_frames = speed_drawer.draw(
                output_frames,
                player_tracks_list,
                distances,
                speeds,
            )
        except Exception as e:
            print(f"[WARN] Speed/Distance overlay skipped: {e}")

    if args.draw_passes_interceptions:
        try:
            from drawers import PassInterceptionDrawer

            pi_drawer = PassInterceptionDrawer()
            output_frames = pi_drawer.draw(output_frames, passes, interceptions)
        except Exception as e:
            print(f"[WARN] Pass/Interception overlay skipped: {e}")

    if args.draw_team_control:
        try:
            from drawers import TeamBallControlDrawer

            team_drawer = TeamBallControlDrawer()
            output_frames = team_drawer.draw(
                output_frames, player_assignments, ball_acquisition
            )
        except Exception as e:
            print(f"[WARN] Team control overlay skipped: {e}")

    # Salvataggio video
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    save_video_with_fps(output_frames, output_video_path, fps)
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
    # Keypoints validation toggle
    parser.add_argument(
        "--no_kp_validate",
        action="store_true",
        help="Disabilita la validazione proporzionale dei keypoint del campo",
    )
    # Court keypoints detector params
    parser.add_argument(
        "--court_kp_conf",
        type=float,
        default=0.5,
        help="Soglia conf per keypoints del campo",
    )
    parser.add_argument(
        "--court_kp_batch",
        type=int,
        default=20,
        help="Batch size per keypoints del campo",
    )
    # Player tracker params
    parser.add_argument(
        "--player_imgsz",
        type=int,
        default=None,
        help="imgsz per tracking giocatori (YOLO)",
    )
    parser.add_argument(
        "--player_conf",
        type=float,
        default=0.15,
        help="conf soglia per tracking giocatori",
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
    # BallTracker ROI/tile avanzati
    parser.add_argument(
        "--ball_roi_scale",
        type=float,
        default=2.4,
        help="Fattore di scala della ROI attorno all'ultima bbox",
    )
    parser.add_argument(
        "--ball_roi_limit",
        type=int,
        default=1280,
        help="imgsz massimo per inferenza sulla ROI",
    )
    parser.add_argument(
        "--ball_roi_conf_factor",
        type=float,
        default=0.6,
        help="Fattore moltiplicativo sulla min_conf nella ROI",
    )
    parser.add_argument(
        "--ball_tile_cols",
        type=int,
        default=2,
        help="Numero di colonne della griglia tile",
    )
    parser.add_argument(
        "--ball_tile_rows",
        type=int,
        default=2,
        help="Numero di righe della griglia tile",
    )
    parser.add_argument(
        "--ball_tile_overlap", type=float, default=0.25, help="Overlap fra tile (0..1)"
    )
    parser.add_argument(
        "--ball_tile_imgsz", type=int, default=960, help="imgsz per ciascun tile"
    )
    parser.add_argument(
        "--ball_tile_conf",
        type=float,
        default=0.03,
        help="soglia conf per inferenza sui tile",
    )
    parser.add_argument(
        "--ball_tile_max_det",
        type=int,
        default=1000,
        help="max det per inferenza sui tile",
    )
    parser.add_argument(
        "--ball_no_adaptive",
        action="store_true",
        help="Disabilita la logica adattiva di ricerca/thresholding nel BallTracker",
    )
    # Ball trail overlay
    parser.add_argument(
        "--no_ball_trail",
        action="store_true",
        help="Disabilita la scia della palla disegnata sui frame",
    )
    parser.add_argument(
        "--ball_trail_len",
        type=int,
        default=25,
        help="Lunghezza massima della scia della palla",
    )
    parser.add_argument(
        "--ball_trail_thickness",
        type=int,
        default=2,
        help="Spessore della polilinea della scia della palla",
    )
    # Shot trajectory overlay
    parser.add_argument(
        "--no_shot_trajectory",
        action="store_true",
        help="Disabilita la traiettoria del tiro (linea verde)",
    )
    # Frame number overlay
    parser.add_argument(
        "--draw_frame_numbers",
        action="store_true",
        help="Disegna l'indice di frame in alto a sinistra",
    )
    # Ball possession thresholds
    parser.add_argument(
        "--possession_threshold",
        type=int,
        default=50,
        help="Distanza max (px) per assegnare il possesso se containment basso",
    )
    parser.add_argument(
        "--pos_min_frames",
        type=int,
        default=11,
        help="Minimo numero di frame consecutivi per confermare possesso",
    )
    parser.add_argument(
        "--pos_containment_threshold",
        type=float,
        default=0.8,
        help="Soglia di containment della palla nella bbox giocatore per possesso immediato",
    )
    # TeamAssigner opts
    parser.add_argument(
        "--team_assigner_half",
        action="store_true",
        help="Usa FP16 per il modello CLIP del TeamAssigner (solo GPU)",
    )
    parser.add_argument(
        "--team_assigner_stride",
        type=int,
        default=1,
        help="Ricalcola l'assegnazione squadra ogni N frame (1=ogni frame)",
    )
    parser.add_argument(
        "--team_assigner_batch",
        action="store_true",
        help="Esegue l'inferenza CLIP in batch sui giocatori del frame",
    )
    # Overlays
    parser.add_argument(
        "--draw_speed_distance",
        action="store_true",
        help="Disegna velocità e distanza dei giocatori",
    )
    parser.add_argument(
        "--draw_passes_interceptions",
        action="store_true",
        help="Disegna overlay con conteggio passaggi e intercetti",
    )
    parser.add_argument(
        "--draw_team_control",
        action="store_true",
        help="Disegna overlay con controllo palla per squadra",
    )
    args = parser.parse_args()
    main(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        read_stubs=not args.no_stubs,
    )
