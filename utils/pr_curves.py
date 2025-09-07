import os
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np


def pr_from_detection_scores(
    detection_scores: Iterable[dict], total_gt: int, smooth: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a precision-recall curve from detection scores using the standard
    object-detection protocol (sort predictions by score desc; accumulate TP/FP;
    recall is TP/TotalGT; precision is TP/(TP+FP)).

    Args:
        detection_scores: iterable of {"score": float, "match": 0|1} for every
            predicted detection considered during evaluation.
        total_gt: number of ground truth objects (for the class) in the dataset/video.
        smooth: if True, apply precision envelope (monotonic non-increasing precision
            by taking cummax from right to left), which stabilizes the curve and makes
            it consistent with AP computation by interpolation.

    Returns:
        recall: np.ndarray in [0,1]
        precision: np.ndarray in [0,1]
    """
    if total_gt <= 0:
        return np.array([0.0]), np.array([1.0])

    # Sort predictions by score descending
    dets: List[dict] = sorted(
        (d for d in detection_scores if d is not None and "score" in d and "match" in d),
        key=lambda x: float(x["score"]),
        reverse=True,
    )
    tp = 0
    fp = 0
    recalls: List[float] = []
    precisions: List[float] = []

    for d in dets:
        if int(d.get("match", 0)) == 1:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / float(total_gt))
        precisions.append(tp / float(tp + fp))

    if not recalls:
        return np.array([0.0]), np.array([1.0])

    recall_arr = np.array(recalls, dtype=float)
    precision_arr = np.array(precisions, dtype=float)

    if smooth and precision_arr.size > 0:
        # Precision envelope: enforce non-increasing precision when recall increases
        # by taking cumulative maximum from right to left
        precision_arr = np.maximum.accumulate(precision_arr[::-1])[::-1]

    # Ensure the curve has at least two points for plotting
    if recall_arr.size == 0:
        recall_arr = np.array([0.0])
        precision_arr = np.array([1.0])
    if recall_arr.size == 1:
        # duplicate last point at recall=1.0 with same precision to make a visible line
        recall_arr = np.array([0.0, 1.0]) if recall_arr[0] == 0.0 else np.array([0.0, recall_arr[0]])
        precision_arr = np.array([precision_arr[0], precision_arr[0]])
    elif recall_arr[0] > 0.0:
        # Prepend origin if needed for nicer plots
        recall_arr = np.insert(recall_arr, 0, 0.0)
        precision_arr = np.insert(precision_arr, 0, precision_arr[0])

    # Clip to [0,1]
    recall_arr = np.clip(recall_arr, 0.0, 1.0)
    precision_arr = np.clip(precision_arr, 0.0, 1.0)
    return recall_arr, precision_arr


def pr_with_thresholds(
    detection_scores: Iterable[dict], total_gt: int, smooth: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like pr_from_detection_scores but also returns the threshold sequence.

    Thresholds are taken as the score of the last included prediction at each step
    (i.e., the k-th highest score when considering the first k predictions).
    """
    if total_gt <= 0:
        return np.array([0.0]), np.array([1.0]), np.array([1.0])

    dets: List[dict] = sorted(
        (d for d in detection_scores if d is not None and "score" in d and "match" in d),
        key=lambda x: float(x["score"]),
        reverse=True,
    )
    tp = 0
    fp = 0
    recalls: List[float] = []
    precisions: List[float] = []
    thresholds: List[float] = []
    for d in dets:
        if int(d.get("match", 0)) == 1:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / float(total_gt))
        precisions.append(tp / float(tp + fp))
        thresholds.append(float(d.get("score", 0.0)))

    recall_arr = np.array(recalls, dtype=float)
    precision_arr = np.array(precisions, dtype=float)
    thr_arr = np.array(thresholds, dtype=float)

    if smooth and precision_arr.size > 0:
        precision_arr = np.maximum.accumulate(precision_arr[::-1])[::-1]

    if recall_arr.size == 0:
        recall_arr = np.array([0.0])
        precision_arr = np.array([1.0])
        thr_arr = np.array([1.0])
    if recall_arr.size == 1:
        recall_arr = np.array([0.0, 1.0]) if recall_arr[0] == 0.0 else np.array([0.0, recall_arr[0]])
        precision_arr = np.array([precision_arr[0], precision_arr[0]])
        thr_arr = np.array([thr_arr[0], thr_arr[0]])
    elif recall_arr[0] > 0.0:
        recall_arr = np.insert(recall_arr, 0, 0.0)
        precision_arr = np.insert(precision_arr, 0, precision_arr[0])
        thr_arr = np.insert(thr_arr, 0, thr_arr[0])
    return recall_arr, precision_arr, thr_arr


def summarize_best_f1(
    detection_scores: Iterable[dict], total_gt: int
) -> Dict[str, Any]:
    """Return best-F1 summary scanning thresholds over sorted predictions.

    Returns keys: best_f1, best_precision, best_recall, best_threshold.
    """
    if total_gt <= 0:
        return {
            "best_f1": 0.0,
            "best_precision": 1.0,
            "best_recall": 0.0,
            "best_threshold": None,
        }
    dets: List[dict] = sorted(
        (d for d in detection_scores if d is not None and "score" in d and "match" in d),
        key=lambda x: float(x["score"]),
        reverse=True,
    )
    tp = 0
    fp = 0
    best = {
        "best_f1": 0.0,
        "best_precision": 1.0,
        "best_recall": 0.0,
        "best_threshold": None,
    }
    for d in dets:
        if int(d.get("match", 0)) == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / float(tp + fp)
        rec = tp / float(total_gt)
        denom = (prec + rec)
        f1 = 2 * (prec * rec) / denom if denom > 0 else 0.0
        if f1 >= best["best_f1"]:
            best = {
                "best_f1": float(f1),
                "best_precision": float(prec),
                "best_recall": float(rec),
                "best_threshold": float(d.get("score", 0.0)),
            }
    return best


def save_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    out_prefix: str,
    title: str | None = None,
    thresholds: np.ndarray | None = None,
) -> None:
    """
    Save PR data to CSV and, if matplotlib is available, also PNG/PDF images.

    Args:
        recall: array of recall values [0..1]
        precision: array of precision values [0..1]
        out_prefix: path prefix without extension (e.g., path/to/my_video_ball)
        title: optional plot title
    """
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = f"{out_prefix}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        if thresholds is not None and len(thresholds) == len(recall):
            f.write("recall,precision,threshold\n")
            for r, p, t in zip(recall, precision, thresholds):
                f.write(f"{float(r):.6f},{float(p):.6f},{float(t):.6f}\n")
        else:
            f.write("recall,precision\n")
            for r, p in zip(recall, precision):
                f.write(f"{float(r):.6f},{float(p):.6f}\n")

    # Figures (best-effort)
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(recall, precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        if title:
            plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = f"{out_prefix}.png"
        pdf_path = f"{out_prefix}.pdf"
        plt.savefig(png_path, dpi=200)
        plt.savefig(pdf_path)
        plt.close()
    except Exception:
        # Matplotlib may be missing; CSV is still written.
        pass


def ap_trapezoid(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the PR curve via trapezoidal rule (expects smoothed PR)."""
    if recall is None or precision is None or len(recall) == 0:
        return 0.0
    r = np.asarray(recall, dtype=float)
    p = np.asarray(precision, dtype=float)
    # Ensure sorted by recall
    order = np.argsort(r)
    r = r[order]
    p = p[order]
    area = float(np.trapz(p, r))
    return max(0.0, min(1.0, area))


def precision_at_recall_targets(
    recall: np.ndarray, precision: np.ndarray, targets: List[float]
) -> List[float]:
    out: List[float] = []
    r = np.asarray(recall, dtype=float)
    p = np.asarray(precision, dtype=float)
    for t in targets:
        mask = r >= float(t)
        out.append(float(np.max(p[mask])) if np.any(mask) else 0.0)
    return out


def recall_at_precision_targets(
    recall: np.ndarray, precision: np.ndarray, targets: List[float]
) -> List[float]:
    out: List[float] = []
    r = np.asarray(recall, dtype=float)
    p = np.asarray(precision, dtype=float)
    for t in targets:
        mask = p >= float(t)
        out.append(float(np.max(r[mask])) if np.any(mask) else 0.0)
    return out
