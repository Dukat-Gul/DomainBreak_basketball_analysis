"""PR-curve maker

Usage examples:

1) Partendo dal report JSON esportato da evaluation.py:
   python scripts/make_pr_curve.py --report-json debug_output/report_video.json \
       --out-dir figures/pr_curves

2) In alternativa, passare detection_scores raw (lista di {score,match}) e total_gt:
   python scripts/make_pr_curve.py --scores-json path/to/scores.json --total-gt 123 \
       --out-dir figures/pr_curves --label "My Video"

Se matplotlib è disponibile, salva anche PNG/PDF; altrimenti salva solo CSV.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

# Ensure project root (one level up from scripts/) is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.pr_curves import pr_from_detection_scores, pr_with_thresholds, save_pr_curve


def _load_from_report(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label = data.get("video", os.path.splitext(os.path.basename(path))[0])
    # Preferisci PR già precalcolata; altrimenti ricalcola dalle detection
    pr = data.get("ball", {}).get("pr_curve") or data.get("pr_curve")
    if pr and "recall" in pr and "precision" in pr:
        recall = np.array(pr["recall"], dtype=float)
        precision = np.array(pr["precision"], dtype=float)
        thresholds = (
            np.array(pr.get("thresholds", []), dtype=float)
            if pr.get("thresholds") is not None
            else None
        )
        return recall, precision, thresholds, label
    # Fallback: costruisci dai detection_scores
    det = data.get("ball", {}).get("metrics", {})
    detection_scores = data.get("ball", {}).get("detection_scores", []) or data.get(
        "detection_scores", []
    )
    total_gt = (
        int(data.get("ball", {}).get("total_gt", 0))
        or int(det.get("total_ground_truth", 0))
        or int(data.get("total_ground_truth", 0))
    )
    recall, precision, thresholds = pr_with_thresholds(detection_scores, total_gt)
    return recall, precision, thresholds, label


def _load_from_scores(
    path: str, total_gt: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accetta sia formato {"detection_scores": [...]} sia lista nuda
    detection_scores = data.get("detection_scores", data)
    recall, precision, thresholds = pr_with_thresholds(detection_scores, int(total_gt))
    label = os.path.splitext(os.path.basename(path))[0]
    return recall, precision, thresholds, label


def main():
    parser = argparse.ArgumentParser(description="Crea curve Precision-Recall")
    parser.add_argument(
        "--report-json",
        nargs="+",
        default=None,
        help="Uno o più file JSON prodotti da evaluation.py (--report_json)",
    )
    parser.add_argument(
        "--scores-json",
        nargs="+",
        default=None,
        help="Uno o più JSON con detection_scores (lista di {score,match})",
    )
    parser.add_argument(
        "--total-gt",
        type=int,
        default=None,
        help="Totale GT necessari se si usa --scores-json senza report",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures/pr_curves",
        help="Cartella di output per CSV/figure",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Etichetta manuale da usare per --scores-json (un solo file)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    items: List[Tuple[np.ndarray, np.ndarray, np.ndarray | None, str]] = []
    if args.report_json:
        for p in args.report_json:
            r, p_curve, thr, lab = _load_from_report(p)
            items.append((r, p_curve, thr, lab))
    if args.scores_json:
        if args.total_gt is None:
            raise SystemExit("--total-gt è obbligatorio quando si usa --scores-json")
        for idx, pth in enumerate(args.scores_json):
            r, p_curve, thr, lab = _load_from_scores(pth, args.total_gt)
            if args.label and len(args.scores_json) == 1 and idx == 0:
                lab = args.label
            items.append((r, p_curve, thr, lab))

    if not items:
        raise SystemExit("Nessun input valido. Usa --report-json o --scores-json.")

    # Salva una curva per elemento
    for recall, precision, thresholds, label in items:
        stem = label.replace(" ", "_")
        out_prefix = os.path.join(args.out_dir, stem)
        save_pr_curve(
            recall, precision, out_prefix, title=f"PR — {label}", thresholds=thresholds
        )
        print(f"Salvato: {out_prefix}.csv (+PNG/PDF se matplotlib disponibile)")


if __name__ == "__main__":
    main()
