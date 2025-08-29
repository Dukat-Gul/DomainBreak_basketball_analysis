import argparse
import json
import re
import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import read_video
from configs import PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH


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
            ground_truth[frame_num] = {"ball": [], "player": [], "referee": [], "rim": []}

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
                ground_truth[frame_num] = {"ball": [], "player": [], "referee": [], "rim": []}
            ground_truth[frame_num].setdefault("referee", []).append(bbox)
        elif category_name == "rim":
            x1, y1, w, h = ann["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            if frame_num not in ground_truth:
                ground_truth[frame_num] = {"ball": [], "player": [], "referee": [], "rim": []}
            ground_truth[frame_num].setdefault("rim", []).append(bbox)

    print(f"Ground truth per '{video_name_base}' processato.")
    return ground_truth, gt_width, gt_height


def main(input_video_path, ground_truth_path, gt_resize_mode="stretch", pred_imgsz=960):
    video_filename = os.path.basename(input_video_path)
    print(f"1. Avvio debug visivo per: {video_filename}")

    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Le immagini di debug verranno salvate in: {output_dir}")

    print("2. Caricamento Ground Truth...")
    ground_truth_data, gt_width, gt_height = load_ground_truth(
        ground_truth_path, video_filename
    )
    if not ground_truth_data:
        return

    print("3. Caricamento modelli e analisi frame per frame...")
    all_frames = read_video(input_video_path)
    if not all_frames:
        print("Errore: impossibile leggere i frame dal video.")
        return

    video_height, video_width, _ = all_frames[0].shape
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
        if gt_resize_mode == "stretch":
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
            # Calcola la scala mantenendo l'aspect ratio (letterboxing)
            scale = min(gt_width / video_width, gt_height / video_height)
            scaled_w, scaled_h = int(video_width * scale), int(video_height * scale)
            # Calcola il padding (offset)
            pad_x = (gt_width - scaled_w) / 2
            pad_y = (gt_height - scaled_h) / 2

            # Funzione per convertire le coordinate
            def convert_bbox(bbox):
                x1, y1, x2, y2 = bbox
                # Rimuovi il padding e riscala alle dimensioni originali del video
                orig_x1 = (x1 - pad_x) / scale
                orig_y1 = (y1 - pad_y) / scale
                orig_x2 = (x2 - pad_x) / scale
                orig_y2 = (y2 - pad_y) / scale
                return [orig_x1, orig_y1, orig_x2, orig_y2]

    else:
        # Se le dimensioni corrispondono, crea una funzione fittizia che non fa nulla
        def convert_bbox(bbox):
            return bbox

    ball_model = YOLO(BALL_DETECTOR_PATH)
    player_model = YOLO(PLAYER_DETECTOR_PATH)

    frames_to_process_indices = sorted(
        [fn for fn in ground_truth_data.keys() if fn < len(all_frames)]
    )

    if not frames_to_process_indices:
        print(
            "Nessun frame con annotazioni trovato per questo video. Impossibile procedere."
        )
        return

    for frame_idx in frames_to_process_indices:
        frame = all_frames[frame_idx].copy()
        print(f"- Processo il frame {frame_idx}...")

        # --- Ground Truth (VERDE) ---
        gt_annotations = ground_truth_data.get(frame_idx, {})
        gt_players = gt_annotations.get("player", [])
        gt_ball = gt_annotations.get("ball", [])
        gt_ref = gt_annotations.get("referee", [])
        gt_rim = gt_annotations.get("rim", [])

        # Applica la conversione corretta delle coordinate
        converted_gt_players = [convert_bbox(b) for b in gt_players]
        converted_gt_ball = [convert_bbox(b) for b in gt_ball]
        converted_gt_ref = [convert_bbox(b) for b in gt_ref]
        converted_gt_rim = [convert_bbox(b) for b in gt_rim]

        frame = draw_boxes(frame, converted_gt_players, (0, 255, 0), "GT_Player")
        frame = draw_boxes(frame, converted_gt_ball, (0, 255, 0), "GT_Ball")
        frame = draw_boxes(frame, converted_gt_ref, (255, 165, 0), "GT_Referee")
        frame = draw_boxes(frame, converted_gt_rim, (255, 0, 255), "GT_Rim")

        # --- Predizioni del Modello (ROSSO) ---
        clean_for_pred = all_frames[frame_idx]  # usa frame pulito per le predizioni
        player_results = player_model(clean_for_pred, verbose=False, imgsz=pred_imgsz)[0]
        player_classes_ids = [
            k for k, v in player_model.names.items() if "player" in v.lower()
        ]
        pred_players = [
            box.xyxy[0].tolist()
            for box in player_results.boxes
            if int(box.cls) in player_classes_ids
        ]
        frame = draw_boxes(frame, pred_players, (0, 0, 255), "Pred_Player")

        ball_results = ball_model(clean_for_pred, verbose=False, imgsz=pred_imgsz)[0]
        # Filtra la classe 'ball' se presente nel modello
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

        # Rileva anche arbitri e canestro (se presenti nel modello dei giocatori)
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
    args = parser.parse_args()
    main(args.input_video, args.ground_truth_file, args.gt_resize_mode, args.pred_imgsz)
