import argparse
import json
import numpy as np
import re
import os
from ultralytics import YOLO
from trackers import PlayerTracker, BallTracker
from utils import read_video
from configs import PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def load_ground_truth(file_path, video_filename):
    with open(file_path, "r") as f:
        data = json.load(f)

    video_name_base = os.path.splitext(video_filename)[0]
    category_map = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Dimensioni delle immagini GT (post-preprocess)
    gt_width = None
    gt_height = None
    if data.get("images"):
        first_image_info = next(
            (img for img in data["images"] if video_name_base in img["file_name"]),
            None,
        )
        if first_image_info:
            gt_width = first_image_info.get("width")
            gt_height = first_image_info.get("height")

    # Filtra le immagini solo per il video corrente
    image_map = {
        img["id"]: img["file_name"]
        for img in data["images"]
        if video_name_base in img["file_name"]
    }

    if not image_map:
        print(
            f"ATTENZIONE: Nessuna immagine trovata nel file di annotazioni per il video '{video_name_base}'"
        )
        return {}, None, None

    ground_truth = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in image_map:
            continue

        file_name = image_map[ann["image_id"]]

        # Estrae il numero del frame (es. da "MeloCrazy3Shot_frame_216_jpg...")
        frame_num_match = re.search(r"frame_(\d+)", file_name)
        if not frame_num_match:
            continue

        frame_num = int(frame_num_match.group(1))

        if frame_num not in ground_truth:
            ground_truth[frame_num] = {"ball": [], "player": []}

        category_name = category_map.get(ann["category_id"])

        # Le etichette del ground truth sono molto specifiche (es. 'player-dribble').
        # Per valutare un rilevatore di giocatori generico, unifichiamo tutte le etichette
        # che contengono la parola 'player' in un'unica categoria.
        unified_category = None
        if category_name == "ball":
            unified_category = "ball"
        elif "player" in category_name:  # Cattura 'player', 'players', 'player-dribble', etc.
            unified_category = "player"

        if unified_category:
            x1, y1, w, h = ann["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            ground_truth[frame_num][unified_category].append(bbox)

    print(f"Ground truth filtrato per '{video_name_base}' e processato con successo.")
    return ground_truth, gt_width, gt_height


def evaluate_detections(
    gt_frames, pred_tracks, class_name, model_class_map, iou_threshold=0.5
):
    tp, fp, fn = 0, 0, 0
    localization_errors = []
    detection_scores = []

    # Trova l'ID numerico per la classe di interesse, gestendo plurali e maiuscole/minuscole
    # es. Cerca "player" e trova "Players" o "players" nel modello
    normalized_class_name = class_name.lower().rstrip("s")
    target_class_id = [
        k
        for k, v in model_class_map.items()
        if v.lower().rstrip("s") == normalized_class_name
    ]

    if not target_class_id:
        print(
            f"Attenzione: classe '{class_name}' (o varianti come plurale/singolare) non trovata nel modello."
        )
        return {}
    target_class_id = target_class_id[0]

    for frame_num, gt_data in gt_frames.items():
        gt_boxes = gt_data.get(class_name, [])
        pred_frame_tracks = pred_tracks.get(frame_num, {})

        pred_boxes_with_scores = []
        for track_id, info in pred_frame_tracks.items():
            if info.get("class_id") == target_class_id:
                pred_boxes_with_scores.append((info["bbox"], info.get("score", 1.0)))

        matched_gt_indices = set()
        for pred_box, score in pred_boxes_with_scores:
            best_iou, best_gt_idx = 0, -1
            for i, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, i

            if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
                localization_errors.append(1 - best_iou)
                detection_scores.append({"score": score, "match": 1})
            else:
                fp += 1
                detection_scores.append({"score": score, "match": 0})

        fn += len(gt_boxes) - len(matched_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    avg_localization_error = np.mean(localization_errors) if localization_errors else 0
    total_gt = tp + fn

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "average_localization_error": avg_localization_error,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_ground_truth": total_gt,
        "detection_scores": detection_scores,
    }


def calculate_average_precision(detection_scores, total_gt):
    if total_gt == 0 or not detection_scores:
        return 0.0
    detection_scores.sort(key=lambda x: x["score"], reverse=True)
    tp_cumulative, fp_cumulative = 0, 0
    recalls, precisions = [], []
    for det in detection_scores:
        if det["match"] == 1:
            tp_cumulative += 1
        else:
            fp_cumulative += 1
        recalls.append(tp_cumulative / total_gt)
        precisions.append(tp_cumulative / (tp_cumulative + fp_cumulative))
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        precision_at_recall = max(
            [p for p, r in zip(precisions, recalls) if r >= t] + [0]
        )
        ap += precision_at_recall
    return ap / 11


def main(input_video_path, ground_truth_path, gt_resize_mode="stretch"):
    video_filename = os.path.basename(input_video_path)
    print(f"1. Analisi del video: {video_filename}")

    print("2. Caricamento e parsing del Ground Truth...")
    ground_truth_data, gt_width, gt_height = load_ground_truth(
        ground_truth_path, video_filename
    )
    if not ground_truth_data:
        return

    # --- NUOVA LOGICA ---
    # Eseguiamo i tracker sull'INTERA sequenza di frame per mantenere la coerenza temporale, 
    # altrimenti i tracker che si basano sulla sequenzialità dei frame fallirebbero. 
    print("3. Esecuzione dei tracker sull\'intero video per coerenza temporale...") 
    all_frames = read_video(input_video_path) 
    if not all_frames:
        print("Errore: impossibile leggere i frame dal video.")
        return

    video_h, video_w, _ = all_frames[0].shape
    print(f"-> Risoluzione Video: {video_w}x{video_h}")
    print(f"-> Risoluzione Annotazioni: {gt_width}x{gt_height}")

    # Converte le bbox del GT nello stesso sistema di coordinate dei frame video
    def identity(b):
        return b

    if gt_width and gt_height and (gt_width != video_w or gt_height != video_h):
        if gt_resize_mode == "stretch":
            sx = video_w / gt_width
            sy = video_h / gt_height

            def convert_bbox(b):
                x1, y1, x2, y2 = b
                return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

        else:
            scale = min(gt_width / video_w, gt_height / video_h)
            scaled_w, scaled_h = int(video_w * scale), int(video_h * scale)
            pad_x = (gt_width - scaled_w) / 2
            pad_y = (gt_height - scaled_h) / 2

            def convert_bbox(b):
                x1, y1, x2, y2 = b
                return [
                    (x1 - pad_x) / scale,
                    (y1 - pad_y) / scale,
                    (x2 - pad_x) / scale,
                    (y2 - pad_y) / scale,
                ]
    else:
        convert_bbox = identity

    gt_converted = {}
    for fn, classes in ground_truth_data.items():
        gt_converted[fn] = {
            "ball": [convert_bbox(b) for b in classes.get("ball", [])],
            "player": [convert_bbox(b) for b in classes.get("player", [])],
        }

    # Esecuzione del tracking della palla 
    ball_model = YOLO(BALL_DETECTOR_PATH) 
    ball_tracker = BallTracker(ball_model) 
    all_ball_tracks = ball_tracker.track_frames(all_frames) 

    # Esecuzione del tracking dei giocatori 
    player_model = YOLO(PLAYER_DETECTOR_PATH) 
    player_tracker = PlayerTracker(player_model) 
    all_player_tracks = player_tracker.get_object_tracks(all_frames) 

    # Le funzioni di valutazione confronteranno le predizioni (per tutti i frame)
    # con il ground truth convertito nello spazio dei frame video.

    print("\n--- INIZIO REPORT DI VALUTAZIONE ---") 

    print("\n4. Valutazione del rilevamento della PALLA:")
    # Usa la mappa classi reale del modello palla (single o multi-classe)
    ball_model_map = ball_model.names
    ball_metrics = evaluate_detections(gt_converted, all_ball_tracks, "ball", ball_model_map)
    ball_ap = calculate_average_precision(
        ball_metrics["detection_scores"], ball_metrics["total_ground_truth"]
    )
    print(
        f"  - Precision: {ball_metrics['precision']:.4f}, Recall: {ball_metrics['recall']:.4f}, F1-Score: {ball_metrics['f1_score']:.4f}"
    )
    print(f"  - Average Precision (AP): {ball_ap:.4f}")
    print(
        f"  - TP: {ball_metrics['true_positives']}, FP: {ball_metrics['false_positives']}, FN: {ball_metrics['false_negatives']}"
    )

    print("\n5. Valutazione del tracciamento dei GIOCATORI:")
    # Usiamo la mappa delle classi del modello dei giocatori
    player_model_map = player_model.names
    player_metrics = evaluate_detections(gt_converted, all_player_tracks, "player", player_model_map)
    player_ap = calculate_average_precision(
        player_metrics["detection_scores"], player_metrics["total_ground_truth"]
    )
    print(
        f"  - Precision: {player_metrics['precision']:.4f}, Recall: {player_metrics['recall']:.4f}, F1-Score: {player_metrics['f1_score']:.4f}"
    )
    print(f"  - Average Precision (AP): {player_ap:.4f}")
    print(
        f"  - Errore Medio di Localizzazione (1 - IoU): {player_metrics['average_localization_error']:.4f}"
    )
    print(
        f"  - TP: {player_metrics['true_positives']}, FP: {player_metrics['false_positives']}, FN: {player_metrics['false_negatives']}"
    )

    print("\n--- FINE REPORT ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esegue la valutazione dei tracker rispetto a un file di ground truth."
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
        "--gt_resize_mode",
        type=str,
        default="stretch",
        choices=["stretch", "letterbox"],
        help="Preprocess delle immagini GT in Roboflow: 'stretch' (default) oppure 'letterbox'",
    )
    args = parser.parse_args()
    main(args.input_video, args.ground_truth_file, args.gt_resize_mode)
