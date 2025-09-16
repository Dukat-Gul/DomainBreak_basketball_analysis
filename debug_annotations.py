import argparse
import json
import re
import os
import cv2
import numpy as np
import time
import pandas as pd
import torch
from utils import save_stub
from ultralytics import YOLO
from utils import read_video
from configs import PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH
from trackers import BallTracker


# Funzione per disegnare i bounding box
def draw_boxes(frame, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Posiziona l'etichetta sopra il riquadro
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return frame


# Riusiamo la funzione di caricamento del ground truth da evaluation.py
def load_ground_truth(file_path, video_filename):
    with open(file_path, "r") as f:
        data = json.load(f)

    video_name_base = os.path.splitext(video_filename)[0]

    # Estrai dimensioni dal primo file di immagine trovato, se disponibili
    gt_width, gt_height = None, None
    if data.get("images"):
        first_image_info = next(
            (img for img in data["images"] if video_name_base in img["file_name"]), None
        )
        if first_image_info:
            gt_width = first_image_info.get("width")
            gt_height = first_image_info.get("height")

    category_map = {cat["id"]: cat["name"] for cat in data["categories"]}
    image_map = {
        img["id"]: img["file_name"]
        for img in data["images"]
        if video_name_base in img["file_name"]
    }

    if not image_map:
        print(f"ATTENZIONE: Nessuna immagine trovata per il video '{video_name_base}'")
        return {}, None, None

    ground_truth = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in image_map:
            continue

        file_name = image_map[ann["image_id"]]
        frame_num_match = re.search(r"frame_(\d+)", file_name)
        if not frame_num_match:
            continue

        frame_num = int(frame_num_match.group(1))
        if frame_num not in ground_truth:
            ground_truth[frame_num] = {
                "ball": [],
                "player": [],
                "referee": [],
                "rim": [],
            }

        category_name = category_map.get(ann["category_id"])

        unified_category = None
        if category_name == "ball":
            unified_category = "ball"
        elif "player" in category_name:
            unified_category = "player"

        if unified_category:
            x1, y1, w, h = ann["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            ground_truth[frame_num][unified_category].append(bbox)
        # Aggiungi GT per arbitro e canestro
        elif category_name == "referee":
            x1, y1, w, h = ann["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            # crea struttura se non presente
            if frame_num not in ground_truth:
                ground_truth[frame_num] = {
                    "ball": [],
                    "player": [],
                    "referee": [],
                    "rim": [],
                }
            ground_truth[frame_num].setdefault("referee", []).append(bbox)
        elif category_name == "rim":
            x1, y1, w, h = ann["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            if frame_num not in ground_truth:
                ground_truth[frame_num] = {
                    "ball": [],
                    "player": [],
                    "referee": [],
                    "rim": [],
                }
            ground_truth[frame_num].setdefault("rim", []).append(bbox)

    print(f"Ground truth per '{video_name_base}' processato.")
    return ground_truth, gt_width, gt_height


def main(args):
    # Create stubs directory
    stubs_dir = "stubs"
    os.makedirs(stubs_dir, exist_ok=True)
    ball_tracks_stub = os.path.join(stubs_dir, "debug_ball_track_stubs.pkl")

    video_filename = os.path.basename(args.input_video)
    print(f"1. Avvio debug visivo per: {video_filename}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Le immagini di debug verranno salvate in: {output_dir}")

    print("2. Caricamento Ground Truth...")
    ground_truth_data, gt_width, gt_height = load_ground_truth(
        args.ground_truth_file, video_filename
    )
    if not ground_truth_data:
        return

    print("3. Caricamento modelli e analisi frame per frame...")
    all_frames = read_video(args.input_video)
    if not all_frames:
        print("Errore: impossibile leggere i frame dal video.")
        return

    video_height, video_width, _ = all_frames[0].shape
    total_frames = len(all_frames)
    print(f"-> Risoluzione Video: {video_width}x{video_height}")
    print(f"-> Risoluzione Annotazioni: {gt_width}x{gt_height}")

    if (
        gt_width
        and video_width
        and gt_height
        and video_height
        and (gt_width != video_width or gt_height != video_height)
    ):
        print("\n*** ATTENZIONE: RISOLUZIONE NON CORRISPONDENTE! ***")
        if args.gt_resize_mode == "stretch":
            print(
                "Le immagini GT sono state 'Resize (Stretch)': applico scala anisotropa (sx, sy)."
            )
            sx = video_width / gt_width
            sy = video_height / gt_height

            def convert_bbox(bbox):
                x1, y1, x2, y2 = bbox
                return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

        else:
            print("Modalità 'letterbox': rimuovo padding e divido per la scala.")
            scale = min(gt_width / video_width, gt_height / video_height)
            scaled_w, scaled_h = int(video_width * scale), int(video_height * scale)
            pad_x = (gt_width - scaled_w) / 2
            pad_y = (gt_height - scaled_h) / 2

            def convert_bbox(bbox):
                x1, y1, x2, y2 = bbox
                orig_x1 = (x1 - pad_x) / scale
                orig_y1 = (y1 - pad_y) / scale
                orig_x2 = (x2 - pad_x) / scale
                orig_y2 = (y2 - pad_y) / scale
                return [orig_x1, orig_y1, orig_x2, orig_y2]

    else:

        def convert_bbox(bbox):
            return bbox

    # Risoluzione del device
    req_device = (args.device or "auto").strip().lower()
    if req_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = args.device
    print(f"4. Uso del device per YOLO: {resolved_device}")

    ball_model = YOLO(BALL_DETECTOR_PATH)
    player_model = YOLO(PLAYER_DETECTOR_PATH)
    # Prova a spostare i modelli sul device richiesto
    try:
        ball_model.to(resolved_device)
        player_model.to(resolved_device)
    except Exception as e:
        print(
            f"[WARN] Impossibile impostare il device '{resolved_device}' (errore: {e}). Fallback a CPU."
        )
        resolved_device = "cpu"
    ball_tracker = BallTracker(
        ball_model,
        imgsz=args.pred_imgsz,
        min_conf=args.ball_conf_min,
        max_search_radius=args.ball_search_radius,
        area_ratio_min=args.area_ratio_min,
        area_ratio_max=args.area_ratio_max,
        use_tta=args.use_tta,
        w_conf=args.w_conf,
        w_orange=args.w_orange,
        w_dist=args.w_dist,
        debug_ball=args.debug_ball,
    )
    ball_tracker.tracker.max_misses = args.ball_max_misses
    # Applica toggle/parametri per A/B testing del tracker palla
    if hasattr(args, "no_adaptive") and args.no_adaptive:
        ball_tracker.adaptive = False
    if hasattr(args, "no_player_prox") and args.no_player_prox:
        ball_tracker.enable_player_proximity = False
    if getattr(args, "w_player_prox", None) is not None:
        ball_tracker.w_player_prox = float(args.w_player_prox)
    if getattr(args, "player_min_conf", None) is not None:
        ball_tracker.player_min_conf = float(args.player_min_conf)
    if getattr(args, "prox_r_frac", None) is not None:
        ball_tracker.prox_r_frac = float(args.prox_r_frac)
    if getattr(args, "w_iou", None) is not None:
        ball_tracker.w_iou = float(args.w_iou)
    if getattr(args, "min_fused_select", None) is not None:
        ball_tracker.min_fused_select = float(args.min_fused_select)
    all_ball_tracks = ball_tracker.track_frames(all_frames)
    # Esporta metriche se richiesto
    if getattr(args, "metrics_csv", None):
        ball_tracker.export_metrics_csv(args.metrics_csv)
        print(f"[Metrics] CSV salvato in: {args.metrics_csv}")
    if getattr(args, "metrics_json", None):
        ball_tracker.export_metrics_json(args.metrics_json)
        print(f"[Metrics] JSON salvato in: {args.metrics_json}")

    # Logica per selezionare i frame da processare
    gt_frames_with_ball = sorted(
        [fn for fn, d in ground_truth_data.items() if d.get("ball")]
    )
    sel = set()
    for fn in gt_frames_with_ball:
        a = max(0, fn - args.window)
        b = min(total_frames - 1, fn + args.window)
        for k in range(a, b + 1):
            sel.add(k)

    frames_to_process_indices = sorted(i for i in sel if (i % args.frame_stride) == 0)

    print(
        f"\nSelezionati {len(frames_to_process_indices)} frame su {total_frames} totali "
        f"(window: {args.window}, stride: {args.frame_stride})."
    )

    if not frames_to_process_indices:
        print(
            "Nessun frame con annotazioni trovato per questo video. Impossibile procedere."
        )
        return

    for frame_idx in frames_to_process_indices:
        if frame_idx >= len(all_frames):
            continue
        frame = all_frames[frame_idx].copy()
        print(f"- Processo il frame {frame_idx}...")

        # --- Ground Truth (VERDE) ---
        gt_annotations = ground_truth_data.get(frame_idx, {})
        gt_players = gt_annotations.get("player", [])
        gt_ball = gt_annotations.get("ball", [])
        gt_ref = gt_annotations.get("referee", [])
        gt_rim = gt_annotations.get("rim", [])

        converted_gt_players = [convert_bbox(b) for b in gt_players]
        converted_gt_ball = [convert_bbox(b) for b in gt_ball]
        converted_gt_ref = [convert_bbox(b) for b in gt_ref]
        converted_gt_rim = [convert_bbox(b) for b in gt_rim]

        frame = draw_boxes(frame, converted_gt_players, (0, 255, 0), "GT_Player")
        frame = draw_boxes(frame, converted_gt_ball, (0, 255, 0), "GT_Ball")
        frame = draw_boxes(frame, converted_gt_ref, (255, 165, 0), "GT_Referee")
        frame = draw_boxes(frame, converted_gt_rim, (255, 0, 255), "GT_Rim")

        # --- Predizioni del Modello (ROSSO) ---
        clean_for_pred = all_frames[frame_idx]
        t0 = time.time()
        player_results = player_model(
            clean_for_pred,
            verbose=False,
            imgsz=args.pred_imgsz,
            device=resolved_device,
        )[0]
        player_classes_ids = [
            k for k, v in player_model.names.items() if "player" in v.lower()
        ]
        pred_players = [
            box.xyxy[0].tolist()
            for box in player_results.boxes
            if int(box.cls) in player_classes_ids
        ]
        frame = draw_boxes(frame, pred_players, (0, 0, 255), "Pred_Player")

        ball_results = ball_model(
            clean_for_pred,
            verbose=False,
            imgsz=args.pred_imgsz,
            device=resolved_device,
        )[0]
        dt = max(1e-3, time.time() - t0)
        fps = 1.0 / dt
        try:
            ball_class_id = next(
                k for k, v in ball_model.names.items() if v.lower() == "ball"
            )
        except StopIteration:
            ball_class_id = 0
        pred_ball = [
            box.xyxy[0].tolist()
            for box in ball_results.boxes
            if int(box.cls) == ball_class_id
        ]
        frame = draw_boxes(frame, pred_ball, (0, 0, 255), "Pred_Ball")

        track_frame = all_ball_tracks.get(frame_idx, {})
        track_items = list(track_frame.items())
        track_boxes = [info["bbox"] for _, info in track_items]
        if track_boxes:
            frame = draw_boxes(frame, track_boxes, (0, 255, 255), "Track_Ball")
            try:
                _, info = track_items[0]
                x1, y1, x2, y2 = map(int, info["bbox"])
                meta_str = []
                if "state" in info:
                    meta_str.append(f"{info['state']}")
                if "tracker_misses" in info:
                    meta_str.append(f"miss:{info['tracker_misses']}")
                if "maha" in info and info["maha"] is not None:
                    meta_str.append(f"maha:{info['maha']:.2f}")
                if "score" in info:
                    meta_str.append(f"s:{info['score']:.2f}")
                meta = " | ".join(meta_str)
                if meta:
                    cv2.putText(
                        frame,
                        meta,
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )
            except Exception:
                pass

        ref_ids = [k for k, v in player_model.names.items() if v.lower() == "referee"]
        rim_ids = [k for k, v in player_model.names.items() if v.lower() == "rim"]
        if ref_ids:
            pred_ref = [
                box.xyxy[0].tolist()
                for box in player_results.boxes
                if int(box.cls) in ref_ids
            ]
            frame = draw_boxes(frame, pred_ref, (255, 165, 0), "Pred_Referee")
        if rim_ids:
            pred_rim = [
                box.xyxy[0].tolist()
                for box in player_results.boxes
                if int(box.cls) in rim_ids
            ]
            frame = draw_boxes(frame, pred_rim, (255, 0, 255), "Pred_Rim")

        header = f"frame:{frame_idx} | fps~{fps:.1f}"
        cv2.putText(
            frame, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2
        )

        output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, frame)

    print(f"\nAnalisi completata. Controlla le immagini nella cartella '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esegue un debug visivo delle annotazioni e delle predizioni."
    )
    parser.add_argument(
        "--input_video", type=str, required=True, help="Percorso del video di input."
    )
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Percorso del file di annotazioni COCO JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_output",
        help="Cartella dove salvare le immagini di debug (default: debug_output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Seleziona il device per i modelli YOLO: 'auto' (default), 'cpu', 'cuda', 'cuda:0', ecc.",
    )
    parser.add_argument(
        "--gt_resize_mode",
        type=str,
        default="stretch",
        choices=["stretch", "letterbox"],
        help="Preprocess delle immagini GT in Roboflow: 'stretch' (default) oppure 'letterbox'",
    )
    parser.add_argument(
        "--pred_imgsz",
        type=int,
        default=960,
        help="Dimensione di inferenza YOLO per le predizioni (utile per oggetti piccoli)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Numero di frame prima/dopo i frame con GT palla.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Sottocampionamento dei frame selezionati (>=1).",
    )
    parser.add_argument("--ball_conf_min", type=float, default=0.03)
    parser.add_argument("--ball_search_radius", type=float, default=0.06)
    parser.add_argument("--ball_max_misses", type=int, default=8)
    parser.add_argument("--area_ratio_min", type=float, default=2e-5)
    parser.add_argument("--area_ratio_max", type=float, default=5e-3)
    parser.add_argument("--w_conf", type=float, default=1.0)
    parser.add_argument("--w_orange", type=float, default=0.35)
    parser.add_argument("--w_dist", type=float, default=0.2)
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument(
        "--debug_ball",
        action="store_true",
        help="Stampa parametri adattivi e conteggi durante il tracking palla",
    )
    # Nuovi flag per controllare le funzionalità del BallTracker (A/B testing)
    parser.add_argument(
        "--no_adaptive", action="store_true", help="Disabilita ricerca/soglie adattive"
    )
    parser.add_argument(
        "--no_player_prox",
        action="store_true",
        help="Disabilita il prior di prossimità ai giocatori nello scoring",
    )
    parser.add_argument(
        "--w_player_prox",
        type=float,
        default=None,
        help="Peso del prior di prossimità ai giocatori (override)",
    )
    parser.add_argument(
        "--player_min_conf",
        type=float,
        default=None,
        help="Conf minima per la detection dei giocatori usata dal prior",
    )
    parser.add_argument(
        "--prox_r_frac",
        type=float,
        default=None,
        help="Raggio di prossimità (frazione di max(lato frame))",
    )
    parser.add_argument(
        "--w_iou",
        type=float,
        default=None,
        help="Peso del termine IoU con la bbox precedente nello scoring",
    )
    parser.add_argument(
        "--min_fused_select",
        type=float,
        default=None,
        help="Soglia minima del punteggio fuso per accettare un candidato (trigger fallback)",
    )
    parser.add_argument(
        "--no_stubs",
        action="store_true",
        help="Disabilita l'uso degli stub (ricalcola tutto da zero).",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default=None,
        help="Percorso file CSV per esportare metriche per-frame del tracker",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default=None,
        help="Percorso file JSON per esportare metriche per-frame del tracker",
    )
    args = parser.parse_args()
    main(args)
