import argparse
import json
import numpy as np
import re
import os
from typing import Optional
from ultralytics import YOLO
from trackers import PlayerTracker, BallTracker
from utils import read_video
from configs import PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH
from utils.pr_curves import (
    pr_from_detection_scores,
    pr_with_thresholds,
    summarize_best_f1,
    save_pr_curve,
    ap_trapezoid,
    precision_at_recall_targets,
    recall_at_precision_targets,
)


def resolve_device(user_device: Optional[str] = None) -> str:
    """Resolve device string from user input, defaulting to best available.

    - 'auto' or None -> 'cuda' if available else 'cpu'
    - otherwise returns the provided value (e.g., 'cuda', 'cuda:0', 'cpu')
    """
    if user_device and user_device.lower() != "auto":
        return user_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _contiguous_runs(bits):
    """Return lengths of contiguous 0-runs and 1-runs from a 0/1 list."""
    if not bits:
        return [], []
    zeros, ones = [], []
    curr_val = bits[0]
    curr_len = 1
    for b in bits[1:]:
        if b == curr_val:
            curr_len += 1
        else:
            if curr_val == 0:
                zeros.append(curr_len)
            else:
                ones.append(curr_len)
            curr_val = b
            curr_len = 1
    # flush last
    if curr_val == 0:
        zeros.append(curr_len)
    else:
        ones.append(curr_len)
    return zeros, ones


def compute_ball_continuity_metrics(gt_frames_conv, pred_tracks, iou_threshold=0.5):
    """
    Frame-level continuity metrics relative to GT presence of the ball.
    - coverage: fraction of GT-ball frames with a matched prediction
    - avg_miss_run / max_miss_run: lengths of consecutive misses among GT-ball frames
    - avg_hit_run: average consecutive hits length (stability)
    - first_detection_delay: frames from first GT-ball frame to first correct match
    """
    # Build sorted list of frames where GT has a ball
    gt_ball_frames = sorted([fn for fn, d in gt_frames_conv.items() if d.get("ball")])
    if not gt_ball_frames:
        return {
            "coverage": 0.0,
            "avg_miss_run": 0.0,
            "max_miss_run": 0,
            "avg_hit_run": 0.0,
            "first_detection_delay": None,
        }

    hits = []  # list of 0/1 for each GT-ball frame
    first_idx = gt_ball_frames[0]
    first_hit_index = None

    for fn in gt_ball_frames:
        gt_boxes = gt_frames_conv.get(fn, {}).get("ball", [])
        pred_frame_tracks = pred_tracks.get(fn, {})
        matched = 0
        # consider any predicted ball bbox in this frame
        for _, info in pred_frame_tracks.items():
            bbox = info.get("bbox")
            if not bbox:
                continue
            # match against any GT ball in this frame
            for gt in gt_boxes:
                if calculate_iou(bbox, gt) >= iou_threshold:
                    matched = 1
                    break
            if matched:
                break
        hits.append(matched)
        if matched and first_hit_index is None:
            first_hit_index = fn

    coverage = float(sum(hits)) / float(len(hits)) if hits else 0.0
    miss_runs, hit_runs = _contiguous_runs(hits)
    avg_miss_run = float(np.mean(miss_runs)) if miss_runs else 0.0
    max_miss_run = int(np.max(miss_runs)) if miss_runs else 0
    avg_hit_run = float(np.mean(hit_runs)) if hit_runs else 0.0
    first_detection_delay = (
        int(first_hit_index - first_idx) if first_hit_index is not None else None
    )

    return {
        "coverage": coverage,
        "avg_miss_run": avg_miss_run,
        "max_miss_run": max_miss_run,
        "avg_hit_run": avg_hit_run,
        "first_detection_delay": first_detection_delay,
    }


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
            ground_truth[frame_num] = {
                "ball": [],
                "player": [],
                "referee": [],
                "rim": [],
            }

        category_name = category_map.get(ann["category_id"])

        # Le etichette del ground truth sono molto specifiche (es. 'player-dribble').
        # Per valutare un rilevatore di giocatori generico, unifichiamo tutte le etichette
        # che contengono la parola 'player' in un'unica categoria.
        unified_category = None
        if category_name == "ball":
            unified_category = "ball"
        elif (
            "player" in category_name
        ):  # Cattura 'player', 'players', 'player-dribble', etc.
            unified_category = "player"
        elif category_name == "referee":
            unified_category = "referee"
        elif category_name == "rim":
            unified_category = "rim"

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
    detection_scores = []  # list of {score, match, iou, frame}
    tp_ious = []
    fp_list = []  # [(frame, score)]
    fn_list = []  # [(frame, gt_bbox)]

    # Trova l'ID numerico per la classe di interesse, gestendo plurali e maiuscole/minuscole
    # es. Cerca "player" e trova "Players" o "players" nel modello
    # Trova l'ID numerico per la classe di interesse, gestendo plurali e maiuscole/minuscole
    normalized_class_name = class_name.lower().rstrip("s")
    target_class_id = [
        k
        for k, v in model_class_map.items()
        if normalized_class_name
        in v.lower()  # MODIFICA: cerca se 'ball' è contenuto (matcha 'sports ball')
    ]
    if not target_class_id:
        if normalized_class_name == "ball":
            target_class_id = [32]  # Forza ID per 'sports ball' in COCO
        else:
            print(f"Attenzione: classe '{class_name}' non trovata nel modello.")
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
                tp_ious.append(float(best_iou))
                detection_scores.append(
                    {
                        "score": float(score),
                        "match": 1,
                        "iou": float(best_iou),
                        "frame": frame_num,
                    }
                )
            else:
                fp += 1
                fp_list.append((frame_num, float(score)))
                detection_scores.append(
                    {"score": float(score), "match": 0, "iou": 0.0, "frame": frame_num}
                )

        fn += len(gt_boxes) - len(matched_gt_indices)
        # record unmatched GT as FN details
        for i, gt_box in enumerate(gt_boxes):
            if i not in matched_gt_indices:
                fn_list.append((frame_num, gt_box))

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
        "tp_ious": tp_ious,
        "fp_samples": fp_list,
        "fn_samples": fn_list,
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


def main(args):
    # Parametri CLI
    input_video_path = args.input_video
    ground_truth_path = args.ground_truth_file
    gt_resize_mode = args.gt_resize_mode
    pred_imgsz = args.pred_imgsz
    ball_conf_min = args.ball_conf_min
    ball_search_radius = args.ball_search_radius
    ball_max_misses = args.ball_max_misses
    area_ratio_min = args.area_ratio_min
    area_ratio_max = args.area_ratio_max
    w_conf = args.w_conf
    w_orange = args.w_orange
    w_dist = args.w_dist
    w_shape = args.w_shape
    w_maha = args.w_maha
    chi2_gating = args.chi2_gating
    use_tta = args.use_tta
    video_filename = os.path.basename(input_video_path)
    print(f"1. Analisi del video: {video_filename}")

    print("2. Caricamento e parsing del Ground Truth...")
    ground_truth_data, gt_width, gt_height = load_ground_truth(
        ground_truth_path, video_filename
    )
    if not ground_truth_data:
        return
    # Applica offset opzionale agli indici frame del GT per allineare con il video
    if getattr(args, "gt_frame_offset", 0) != 0:
        try:
            off = int(args.gt_frame_offset)
            ground_truth_data = {int(k) + off: v for k, v in ground_truth_data.items()}
            print(f"[GT] Applicato offset ai frame: {off}")
        except Exception:
            pass

    # --- NUOVA LOGICA ---
    # Eseguiamo i tracker sull'INTERA sequenza di frame per mantenere la coerenza temporale,
    # altrimenti i tracker che si basano sulla sequenzialità dei frame fallirebbero.
    print("3. Esecuzione dei tracker sull'intero video per coerenza temporale...")
    device = resolve_device(getattr(args, "device", "auto"))
    print(f"[Device] Inference device: {device}")
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
            "referee": [convert_bbox(b) for b in classes.get("referee", [])],
            "rim": [convert_bbox(b) for b in classes.get("rim", [])],
        }

    def _print_ball_area_quantiles(gt_conv, vw, vh):
        import numpy as np

        fa = float(vw * vh)
        areas = []
        for _, cls in gt_conv.items():
            for b in cls.get("ball", []):
                x1, y1, x2, y2 = b
                a = max(1.0, (x2 - x1) * (y2 - y1)) / fa
                areas.append(a)
        if not areas:
            print("Nessuna palla nel GT per stimare area.")
            return None
        q5, q50, q95 = np.quantile(np.array(areas), [0.05, 0.50, 0.95])
        a_min = 0.5 * q5
        a_max = 2.0 * q95
        print(f"[Ball area ratios] Q5={q5:.3e}  Q50={q50:.3e}  Q95={q95:.3e}")
        print(
            f" → Suggerimento: --area_ratio_min {a_min:.3e}  --area_ratio_max {a_max:.3e}"
        )
        return a_min, a_max

    _suggest = _print_ball_area_quantiles(gt_converted, video_w, video_h)

    total_frames = len(all_frames)

    # fast-dev solo per la palla (mantiene i giocatori completi se --fast_dev_ball_only)
    if args.fast_dev or getattr(args, "fast_dev_ball_only", False):
        ball_gt_frames = sorted([fn for fn, d in gt_converted.items() if d.get("ball")])
        sel = set()
        for fn in ball_gt_frames:
            a = max(0, fn - args.window)
            b = min(total_frames - 1, fn + args.window)
            for k in range(a, b + 1):
                sel.add(k)
        selected_indices = sorted(i for i in sel if (i % args.frame_stride) == 0)
        print(
            f"[FAST_DEV] Valuterò {len(selected_indices)} frame su {total_frames} totali."
        )
        frames_for_ball = [all_frames[i] for i in selected_indices]
        # Filtra anche il GT solo ai frame selezionati
        gt_converted = {
            fn: gt_converted[fn] for fn in selected_indices if fn in gt_converted
        }
    else:
        selected_indices = list(range(total_frames))
        frames_for_ball = all_frames
        print(f"[FULL] Valuterò tutti i {total_frames} frame per la palla.")

    # --- BALL TRACKER ---
    ball_model = YOLO(BALL_DETECTOR_PATH)
    try:
        ball_model.to(device)
    except Exception:
        pass
    ball_tracker = BallTracker(
        ball_model,
        imgsz=pred_imgsz,
        min_conf=ball_conf_min,
        iou_nms=args.ball_iou_nms,
        max_det=args.ball_max_det,
        agnostic_nms=args.ball_agnostic_nms,
        max_search_radius=ball_search_radius,
        area_ratio_min=area_ratio_min,
        area_ratio_max=area_ratio_max,
        use_tta=use_tta,
        w_conf=w_conf,
        w_orange=w_orange,
        w_dist=w_dist,
        w_shape=w_shape,
        w_maha=w_maha,
        chi2_gating=chi2_gating,
        debug_ball=args.debug_ball,
    )
    # Applica configurazioni aggiuntive per A/B testing
    if getattr(args, "no_adaptive", False):
        ball_tracker.adaptive = False
    if getattr(args, "no_player_prox", False):
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
    ball_tracker.tracker.max_misses = ball_max_misses
    all_ball_tracks = ball_tracker.track_frames(
        frames_for_ball, frame_indices=selected_indices
    )
    # Esporta metriche se richiesto
    if getattr(args, "metrics_csv", None):
        ball_tracker.export_metrics_csv(
            args.metrics_csv, frame_indices=selected_indices
        )
        print(f"[Metrics] CSV salvato in: {args.metrics_csv}")
    if getattr(args, "metrics_json", None):
        ball_tracker.export_metrics_json(
            args.metrics_json, frame_indices=selected_indices
        )
        print(f"[Metrics] JSON salvato in: {args.metrics_json}")

    # --- PLAYER TRACKER (opzionale) ---
    if args.skip_players:
        player_model = None
        all_player_tracks = {}
    else:
        player_model = YOLO(PLAYER_DETECTOR_PATH)
        try:
            player_model.to(device)
        except Exception:
            pass
        player_tracker = PlayerTracker(player_model, imgsz=pred_imgsz)
        if args.fast_dev and not getattr(args, "fast_dev_ball_only", False):
            tmp_tracks = player_tracker.get_object_tracks(frames_for_ball)
            # rimappa agli indici globali
            all_player_tracks = {selected_indices[i]: v for i, v in tmp_tracks.items()}
        else:
            all_player_tracks = player_tracker.get_object_tracks(all_frames)

    print("\n--- INIZIO REPORT DI VALUTAZIONE ---")

    # 4) PALLA
    ball_model_map = ball_model.names
    ball_metrics = evaluate_detections(
        gt_converted,
        all_ball_tracks,
        "ball",
        ball_model_map,
        iou_threshold=args.ball_iou_eval,
    )
    ball_ap = calculate_average_precision(
        ball_metrics["detection_scores"], ball_metrics["total_ground_truth"]
    )
    print(f"\n4. Valutazione del rilevamento della PALLA:")
    print(
        f"  - Precision: {ball_metrics['precision']:.4f}, Recall: {ball_metrics['recall']:.4f}, F1-Score: {ball_metrics['f1_score']:.4f}"
    )
    print(f"  - Average Precision (AP): {ball_ap:.4f}")
    print(
        f"  - TP: {ball_metrics['true_positives']}, FP: {ball_metrics['false_positives']}, FN: {ball_metrics['false_negatives']}"
    )
    # Info aggiuntive di copertura
    gt_ball_frames_count = sum(1 for _, d in gt_converted.items() if d.get("ball"))
    pred_ball_frames_count = sum(
        1 for fn in selected_indices if all_ball_tracks.get(fn)
    )
    print(
        f"  - Frame con GT palla: {gt_ball_frames_count}, frame con predizione palla: {pred_ball_frames_count}"
    )
    # Continuità temporale del tracking palla
    cont_metrics = compute_ball_continuity_metrics(
        gt_converted, all_ball_tracks, iou_threshold=args.ball_iou_eval
    )
    print(
        "  - Coverage (GT-ball frames matched): {:.3f}".format(cont_metrics["coverage"])
    )
    print(
        "  - Miss run: avg={:.2f} max={} (frames), Hit run avg={:.2f}".format(
            cont_metrics["avg_miss_run"],
            cont_metrics["max_miss_run"],
            cont_metrics["avg_hit_run"],
        )
    )
    if cont_metrics["first_detection_delay"] is not None:
        print(
            f"  - First detection delay: {cont_metrics['first_detection_delay']} frame(s) from first GT"
        )

    # PR curve (opzionale)
    pr_recall, pr_precision, pr_thresholds = pr_with_thresholds(
        ball_metrics.get("detection_scores", []),
        ball_metrics.get("total_ground_truth", 0),
        smooth=True,
    )
    best_f1 = summarize_best_f1(
        ball_metrics.get("detection_scores", []),
        ball_metrics.get("total_ground_truth", 0),
    )
    ap_auc = ap_trapezoid(pr_recall, pr_precision)
    p_at_r_targets = [0.3, 0.5, 0.7]
    r_at_p_targets = [0.8, 0.9, 0.95]
    p_at_r = precision_at_recall_targets(pr_recall, pr_precision, p_at_r_targets)
    r_at_p = recall_at_precision_targets(pr_recall, pr_precision, r_at_p_targets)
    print(
        "  - Best F1: {f1:.3f} (P={p:.3f}, R={r:.3f}, thr={t:.3f})".format(
            f1=best_f1["best_f1"],
            p=best_f1["best_precision"],
            r=best_f1["best_recall"],
            t=(
                best_f1["best_threshold"]
                if best_f1["best_threshold"] is not None
                else float("nan")
            ),
        )
    )
    print(
        "  - AP(trapz): {ap:.3f} | P@R[0.3,0.5,0.7]={p30:.2f},{p50:.2f},{p70:.2f} | R@P[0.8,0.9,0.95]={r80:.2f},{r90:.2f},{r95:.2f}".format(
            ap=ap_auc,
            p30=p_at_r[0],
            p50=p_at_r[1],
            p70=p_at_r[2],
            r80=r_at_p[0],
            r90=r_at_p[1],
            r95=r_at_p[2],
        )
    )
    if getattr(args, "make_pr", False):
        out_dir = getattr(args, "pr_out_dir", None) or "debug_output/pr_curves"
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(video_filename)[0]
        out_prefix = os.path.join(out_dir, f"{stem}_ball")
        save_pr_curve(
            pr_recall,
            pr_precision,
            out_prefix,
            title=f"PR Curve — Ball — {stem}",
            thresholds=pr_thresholds,
        )
        print(f"  - PR curve salvata in: {out_prefix}.csv/.png")

    # Breakdown del sorgente selezionato (FULL/ROI/TILE/TM) per diagnosi
    src_counts = {}
    try:
        for m in getattr(ball_tracker, "metrics", []) or []:
            s = m.get("src")
            src_counts[s] = src_counts.get(s, 0) + 1
    except Exception:
        pass
    if src_counts:
        total_m = sum(src_counts.values())
        msg = ", ".join(
            f"{k if k is not None else 'NONE'}: {v} ({v/total_m:.1%})"
            for k, v in sorted(src_counts.items(), key=lambda x: (str(x[0])))
        )
        print(f"  - Selezione sorgente (diagnosi): {msg}")

    # Debug opzionale: salva overlay GT vs pred per alcuni frame per verificare allineamento
    if getattr(args, "dump_overlays", False):
        try:
            import cv2

            os.makedirs(args.overlays_dir, exist_ok=True)
            # Scegli alcuni frame con GT palla
            sample_frames = [
                fn
                for fn in sorted(gt_converted.keys())
                if gt_converted.get(fn, {}).get("ball")
            ]
            if args.fast_dev or getattr(args, "fast_dev_ball_only", False):
                # Limita ai frame selezionati in fast-dev
                selset = set(selected_indices)
                sample_frames = [fn for fn in sample_frames if fn in selset]
            sample_frames = sample_frames[: int(getattr(args, "max_overlays", 10))]

            for fn in sample_frames:
                frame = all_frames[fn].copy()
                gt_boxes = gt_converted.get(fn, {}).get("ball", [])
                pred_boxes = []
                for _, info in all_ball_tracks.get(fn, {}).items():
                    if info.get("class_id") is not None:
                        pred_boxes.append(
                            (info.get("bbox"), float(info.get("score", 0.0)))
                        )

                # Disegna GT (verde) e Pred (rosso)
                for x1, y1, x2, y2 in gt_boxes:
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                best_iou = 0.0
                for pb, sc in pred_boxes:
                    x1, y1, x2, y2 = pb
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        2,
                    )
                    for gt in gt_boxes:
                        iou = calculate_iou(pb, gt)
                        if iou > best_iou:
                            best_iou = iou
                txt = f"frame={fn} bestIoU={best_iou:.3f} preds={len(pred_boxes)} gts={len(gt_boxes)}"
                cv2.putText(
                    frame,
                    txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                out_path = os.path.join(args.overlays_dir, f"overlay_{fn}.jpg")
                cv2.imwrite(out_path, frame)
                print(f"[Overlay] Salvato {out_path} — {txt}")
        except Exception as e:
            print(f"[Overlay] Errore creazione overlay: {e}")

    # 5) GIOCATORI (solo se non saltati)
    if not args.skip_players:
        player_model_map = player_model.names
        player_metrics = evaluate_detections(
            gt_converted, all_player_tracks, "player", player_model_map
        )
        player_ap = calculate_average_precision(
            player_metrics["detection_scores"], player_metrics["total_ground_truth"]
        )
        print(f"\n5. Valutazione del tracciamento dei GIOCATORI:")
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
        # PR/Best-F1/AP(trapz)
        p_recall, p_precision = pr_from_detection_scores(
            player_metrics.get("detection_scores", []),
            player_metrics.get("total_ground_truth", 0),
            smooth=True,
        )
        p_best_f1 = summarize_best_f1(
            player_metrics.get("detection_scores", []),
            player_metrics.get("total_ground_truth", 0),
        )
        p_ap_auc = ap_trapezoid(p_recall, p_precision)
        print(
            "  - Best F1 (players): {f1:.3f} (P={p:.3f}, R={r:.3f}) | AP(trapz)={ap:.3f}".format(
                f1=p_best_f1["best_f1"],
                p=p_best_f1["best_precision"],
                r=p_best_f1["best_recall"],
                ap=p_ap_auc,
            )
        )
        # PR curve giocatori (opzionale)
        if getattr(args, "make_pr_players", False):
            out_dir = getattr(args, "pr_out_dir", None) or "debug_output/pr_curves"
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(video_filename)[0]
            out_prefix = os.path.join(out_dir, f"{stem}_players")
            save_pr_curve(
                p_recall,
                p_precision,
                out_prefix,
                title=f"PR Curve — Players — {stem}",
            )
            print(f"  - PR curve (players) salvata in: {out_prefix}.csv/.png")

        # 6–7) ARBITRO e CANESTRO (solo se richiesti)
        if not args.skip_ref_rim:
            referee_metrics = evaluate_detections(
                gt_converted, all_player_tracks, "referee", player_model_map
            )
            referee_ap = calculate_average_precision(
                referee_metrics["detection_scores"],
                referee_metrics["total_ground_truth"],
            )
            print(f"\n6. Valutazione rilevamento ARBITRO:")
            print(
                f"  - Precision: {referee_metrics['precision']:.4f}, Recall: {referee_metrics['recall']:.4f}, F1-Score: {referee_metrics['f1_score']:.4f}"
            )
            print(f"  - Average Precision (AP): {referee_ap:.4f}")
            print(
                f"  - TP: {referee_metrics['true_positives']}, FP: {referee_metrics['false_positives']}, FN: {referee_metrics['false_negatives']}"
            )

            rim_metrics = evaluate_detections(
                gt_converted, all_player_tracks, "rim", player_model_map
            )
            rim_ap = calculate_average_precision(
                rim_metrics["detection_scores"], rim_metrics["total_ground_truth"]
            )
            print(f"\n7. Valutazione rilevamento CANESTRO:")
            print(
                f"  - Precision: {rim_metrics['precision']:.4f}, Recall: {rim_metrics['recall']:.4f}, F1-Score: {rim_metrics['f1_score']:.4f}"
            )
            print(f"  - Average Precision (AP): {rim_ap:.4f}")
            print(
                f"  - TP: {rim_metrics['true_positives']}, FP: {rim_metrics['false_positives']}, FN: {rim_metrics['false_negatives']}"
            )

    # Report JSON opzionale per confronti A/B
    if (
        getattr(args, "metrics_json", None) is None
        and getattr(args, "metrics_csv", None) is None
    ):
        pass  # nothing to export from tracker explicitly
    if getattr(args, "report_json", None):
        report = {
            "video": video_filename,
            "eval": {
                "ball": {
                    **{
                        k: (
                            float(v)
                            if isinstance(v, (int, float, np.floating, np.integer))
                            else v
                        )
                        for k, v in ball_metrics.items()
                        if k != "detection_scores"
                    },
                    "ap": float(ball_ap),
                    "continuity": cont_metrics,
                }
            },
            "tracker_src_breakdown": src_counts,
            "args": {
                "pred_imgsz": pred_imgsz,
                "ball_conf_min": ball_conf_min,
                "ball_search_radius": ball_search_radius,
                "ball_max_misses": ball_max_misses,
                "area_ratio_min": area_ratio_min,
                "area_ratio_max": area_ratio_max,
                "w_conf": w_conf,
                "w_orange": w_orange,
                "w_dist": w_dist,
                "w_shape": w_shape,
                "w_maha": w_maha,
                "chi2_gating": chi2_gating,
                "use_tta": bool(use_tta),
                "ball_iou_eval": float(args.ball_iou_eval),
                "fast_dev": bool(args.fast_dev),
                "fast_dev_ball_only": bool(getattr(args, "fast_dev_ball_only", False)),
                "window": int(args.window),
                "frame_stride": int(args.frame_stride),
                "skip_players": bool(args.skip_players),
            },
        }
        try:
            os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        except Exception:
            pass
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Report] JSON salvato in: {args.report_json}")

    print("\n--- FINE REPORT ---")

    # # Esecuzione del tracking dei giocatori
    # player_model = YOLO(PLAYER_DETECTOR_PATH)
    # player_tracker = PlayerTracker(player_model, imgsz=pred_imgsz)
    # all_player_tracks = player_tracker.get_object_tracks(all_frames)

    # # Le funzioni di valutazione confronteranno le predizioni (per tutti i frame)
    # # con il ground truth convertito nello spazio dei frame video.

    # print("\n--- INIZIO REPORT DI VALUTAZIONE ---")

    # print("\n4. Valutazione del rilevamento della PALLA:")
    # # Usa la mappa classi reale del modello palla (single o multi-classe)
    # ball_model_map = ball_model.names
    # ball_metrics = evaluate_detections(
    #     gt_converted,
    #     all_ball_tracks,
    #     "ball",
    #     ball_model_map,
    #     iou_threshold=args.ball_iou_eval,
    # )

    # ball_ap = calculate_average_precision(
    #     ball_metrics["detection_scores"], ball_metrics["total_ground_truth"]
    # )
    # print(
    #     f"  - Precision: {ball_metrics['precision']:.4f}, Recall: {ball_metrics['recall']:.4f}, F1-Score: {ball_metrics['f1_score']:.4f}"
    # )
    # print(f"  - Average Precision (AP): {ball_ap:.4f}")
    # print(
    #     f"  - TP: {ball_metrics['true_positives']}, FP: {ball_metrics['false_positives']}, FN: {ball_metrics['false_negatives']}"
    # )

    # print("\n5. Valutazione del tracciamento dei GIOCATORI:")
    # # Usiamo la mappa delle classi del modello dei giocatori
    # player_model_map = player_model.names
    # player_metrics = evaluate_detections(
    #     gt_converted, all_player_tracks, "player", player_model_map
    # )
    # player_ap = calculate_average_precision(
    #     player_metrics["detection_scores"], player_metrics["total_ground_truth"]
    # )
    # print(
    #     f"  - Precision: {player_metrics['precision']:.4f}, Recall: {player_metrics['recall']:.4f}, F1-Score: {player_metrics['f1_score']:.4f}"
    # )
    # print(f"  - Average Precision (AP): {player_ap:.4f}")
    # print(
    #     f"  - Errore Medio di Localizzazione (1 - IoU): {player_metrics['average_localization_error']:.4f}"
    # )
    # print(
    #     f"  - TP: {player_metrics['true_positives']}, FP: {player_metrics['false_positives']}, FN: {player_metrics['false_negatives']}"
    # )

    # # Valutazione ARBITRO
    # print("\n6. Valutazione rilevamento ARBITRO:")
    # referee_metrics = evaluate_detections(
    #     gt_converted, all_player_tracks, "referee", player_model_map
    # )
    # referee_ap = calculate_average_precision(
    #     referee_metrics["detection_scores"], referee_metrics["total_ground_truth"]
    # )
    # print(
    #     f"  - Precision: {referee_metrics['precision']:.4f}, Recall: {referee_metrics['recall']:.4f}, F1-Score: {referee_metrics['f1_score']:.4f}"
    # )
    # print(f"  - Average Precision (AP): {referee_ap:.4f}")
    # print(
    #     f"  - TP: {referee_metrics['true_positives']}, FP: {referee_metrics['false_positives']}, FN: {referee_metrics['false_negatives']}"
    # )

    # # Valutazione CANESTRO
    # print("\n7. Valutazione rilevamento CANESTRO:")
    # rim_metrics = evaluate_detections(
    #     gt_converted, all_player_tracks, "rim", player_model_map
    # )
    # rim_ap = calculate_average_precision(
    #     rim_metrics["detection_scores"], rim_metrics["total_ground_truth"]
    # )
    # print(
    #     f"  - Precision: {rim_metrics['precision']:.4f}, Recall: {rim_metrics['recall']:.4f}, F1-Score: {rim_metrics['f1_score']:.4f}"
    # )
    # print(f"  - Average Precision (AP): {rim_ap:.4f}")
    # print(
    #     f"  - TP: {rim_metrics['true_positives']}, FP: {rim_metrics['false_positives']}, FN: {rim_metrics['false_negatives']}"
    # )

    # print("\n--- FINE REPORT ---")

    # Report JSON compatto (opzionale)
    if getattr(args, "report_json", None):
        try:
            rep = {
                "video": video_filename,
                "config": {
                    "iou_eval_ball": args.ball_iou_eval,
                    "pred_imgsz": pred_imgsz,
                },
                "ball": {
                    "metrics": {
                        k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in ball_metrics.items()
                        if k != "detection_scores"
                    },
                    "ap": float(ball_ap),
                    "continuity": cont_metrics,
                    "pr_curve": {
                        "recall": [float(x) for x in pr_recall],
                        "precision": [float(x) for x in pr_precision],
                        "thresholds": [float(x) for x in pr_thresholds],
                    },
                    "best_f1": best_f1,
                    "ap_trapz": float(ap_auc),
                    "precision_at_recall": {
                        str(t): float(v) for t, v in zip(p_at_r_targets, p_at_r)
                    },
                    "recall_at_precision": {
                        str(t): float(v) for t, v in zip(r_at_p_targets, r_at_p)
                    },
                    "total_gt": int(ball_metrics.get("total_ground_truth", 0)),
                    "tp_ious": [float(x) for x in ball_metrics.get("tp_ious", [])],
                    "top_fp_samples": sorted(
                        [
                            {"frame": int(f), "score": float(s)}
                            for f, s in ball_metrics.get("fp_samples", [])
                        ],
                        key=lambda x: x["score"],
                        reverse=True,
                    )[:20],
                    "top_fn_samples": [
                        {"frame": int(f), "gt_bbox": [float(v) for v in b]}
                        for f, b in ball_metrics.get("fn_samples", [])[:20]
                    ],
                },
            }
            if not args.skip_players:
                rep["players"] = {
                    "metrics": {
                        k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in player_metrics.items()
                        if k != "detection_scores"
                    },
                    "ap_voc11": float(player_ap),
                    "pr_curve": {
                        "recall": [float(x) for x in p_recall],
                        "precision": [float(x) for x in p_precision],
                    },
                    "best_f1": p_best_f1,
                    "ap_trapz": float(p_ap_auc),
                }
            os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
            print(f"[Report] Salvato JSON: {args.report_json}")
        except Exception as e:
            print(f"[Report] Errore nel salvataggio del report JSON: {e}")


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
    parser.add_argument(
        "--gt_frame_offset",
        type=int,
        default=0,
        help="Offset intero da sommare agli indici frame del GT per allinearli al video",
    )
    parser.add_argument(
        "--pred_imgsz",
        type=int,
        default=960,
        help="Dimensione di inferenza YOLO (maggiore migliora recall su oggetti piccoli)",
    )
    parser.add_argument(
        "--ball_conf_min",
        type=float,
        default=0.05,
        help="Soglia conf minima per la palla",
    )
    parser.add_argument(
        "--ball_iou_nms",
        type=float,
        default=0.85,
        help="Soglia IoU per NMS della palla (più alto = meno soppressione)",
    )
    parser.add_argument(
        "--ball_max_det",
        type=int,
        default=500,
        help="Max box da tenere dopo NMS per la palla",
    )
    parser.add_argument(
        "--ball_agnostic_nms",
        action="store_true",
        help="Se impostato, usa NMS agnostico alla classe (sconsigliato per palla)",
    )
    parser.add_argument(
        "--ball_search_radius",
        type=float,
        default=0.10,
        help="Raggio di ricerca relativo per associare detections fra frame (0-0.5)",
    )
    parser.add_argument(
        "--ball_max_misses",
        type=int,
        default=5,
        help="Quanti frame mantenere la traccia senza detection (Kalman)",
    )
    parser.add_argument(
        "--area_ratio_min",
        type=float,
        default=2e-5,
        help="Area bbox palla minima come frazione dell'area del frame",
    )
    parser.add_argument(
        "--area_ratio_max",
        type=float,
        default=5e-3,
        help="Area bbox palla massima come frazione dell'area del frame",
    )
    parser.add_argument(
        "--w_conf",
        type=float,
        default=1.0,
        help="Peso della confidenza del detector nello scoring",
    )
    parser.add_argument(
        "--w_orange",
        type=float,
        default=0.35,
        help="Peso della componente cromatica arancione nello scoring",
    )
    parser.add_argument(
        "--w_dist",
        type=float,
        default=0.2,
        help="Peso della vicinanza alla posizione precedente nello scoring",
    )
    parser.add_argument(
        "--w_shape",
        type=float,
        default=0.2,
        help="Peso della circolarità (shape) nello scoring",
    )
    parser.add_argument(
        "--w_maha",
        type=float,
        default=0.4,
        help="Peso (penalità) della distanza di Mahalanobis",
    )
    parser.add_argument(
        "--chi2_gating",
        type=float,
        default=9.21,
        help="Soglia chi^2 (df=2) per il gating di Mahalanobis",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Abilita test-time augmentation per la palla",
    )
    parser.add_argument(
        "--fast_dev", action="store_true", help="Valuta solo un sottoinsieme di frame"
    )
    parser.add_argument(
        "--fast_dev_ball_only",
        action="store_true",
        help="Applica fast-dev solo alla palla; giocatori/referee/rim su tutti i frame",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Numero di frame prima/dopo i frame con GT palla",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Sottocampionamento dei frame selezionati (>=1)",
    )
    parser.add_argument(
        "--skip_players",
        action="store_true",
        help="Salta tracking giocatori per velocizzare",
    )
    parser.add_argument(
        "--skip_ref_rim", action="store_true", help="Salta referee e rim in valutazione"
    )

    parser.add_argument(
        "--ball_iou_eval",
        type=float,
        default=0.50,
        help="Soglia IoU per valutare la palla (tiny objects: 0.30 consigliato)",
    )

    parser.add_argument(
        "--debug_ball",
        action="store_true",
        help="Logga #box e top conf palla per frame",
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
    parser.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="Percorso file JSON per un report compatto di valutazione (precision/recall/F1/AP/continuità)",
    )
    parser.add_argument(
        "--dump_overlays",
        action="store_true",
        help="Salva immagini di debug con GT vs pred per alcuni frame",
    )
    parser.add_argument(
        "--overlays_dir",
        type=str,
        default="debug_output/overlays",
        help="Cartella in cui salvare le overlay di debug",
    )
    parser.add_argument(
        "--max_overlays",
        type=int,
        default=10,
        help="Numero massimo di overlay da salvare",
    )
    # Nuovi flag per A/B testing delle nuove logiche del BallTracker
    parser.add_argument(
        "--no_adaptive",
        action="store_true",
        help="Disabilita la ricerca/soglie adattive del tracker palla",
    )
    parser.add_argument(
        "--no_player_prox",
        action="store_true",
        help="Disabilita il prior di prossimità ai giocatori nello scoring della palla",
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
        "--device",
        type=str,
        default="auto",
        help="Device per l'inferenza YOLO: 'auto', 'cpu', 'cuda', 'cuda:0', ecc.",
    )
    # Opzioni PR curve
    parser.add_argument(
        "--make_pr",
        action="store_true",
        help="Calcola e salva la PR curve per la palla (CSV e, se possibile, PNG/PDF)",
    )
    parser.add_argument(
        "--make_pr_players",
        action="store_true",
        help="Calcola e salva la PR curve per i giocatori",
    )
    parser.add_argument(
        "--pr_out_dir",
        type=str,
        default="debug_output/pr_curves",
        help="Cartella di output per le PR curve",
    )
    args = parser.parse_args()
    main(args)
