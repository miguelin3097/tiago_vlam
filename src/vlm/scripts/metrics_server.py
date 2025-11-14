#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_server.py
Nodo ROS (Noetic) que:
  - Calcula métricas intra-sesión
  - Crea carpeta por sesión
  - Mueve/Copia imágenes
  - Genera y guarda manifest.json
  - Devuelve también el manifest como JSON en la respuesta

Servicio:
  /vlm/finalize_session   (FinalizeSession.srv)
"""

import os
import json
import shutil
import datetime
from pathlib import Path

import rospy
from vlm.srv import FinalizeSession, FinalizeSessionResponse


# ------------------- Normalización y métricas -------------------
def normalize_color(c):
    c = (c or "").strip().lower()
    mapping = {
        "rojo":"red","verde":"green","azul":"blue","amarillo":"yellow",
        "naranja":"orange","morado":"purple","violeta":"purple",
        "rosa":"pink","negro":"black","blanco":"white",
        "marrón":"brown","marron":"brown","cafe":"brown",
        "gris":"gray","grís":"gray"
    }
    return mapping.get(c, c)

def normalize_seq(seq):
    return [normalize_color(s) for s in (seq or [])]

def set_precision_recall_f1(gt, pred):
    s_gt, s_pr = set(gt), set(pred)
    tp = len(s_gt & s_pr)
    prec = (tp / len(s_pr)) if s_pr else 0.0
    rec  = (tp / len(s_gt)) if s_gt else 0.0
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    return prec, rec, f1

def per_position_accuracy(gt, pred):
    n = min(len(gt), len(pred))
    if max(len(gt), len(pred)) == 0:
        return 0.0
    correct = sum(1 for i in range(n) if gt[i] == pred[i])
    penalty = abs(len(gt) - len(pred)) * 0.5
    return max(0.0, (correct - penalty)/max(len(gt), len(pred)))

def hamming_distance(gt, pred):
    n = min(len(gt), len(pred))
    base = sum(1 for i in range(n) if gt[i] != pred[i])
    return base + abs(len(gt) - len(pred))

def footrule_distance(gt, pred):
    r_gt = {c:i for i,c in enumerate(gt)}
    r_pr = {c:i for i,c in enumerate(pred)}
    common = r_gt.keys() & r_pr.keys()
    return sum(abs(r_pr[c] - r_gt[c]) for c in common)

def kendall_pairwise_accuracy(gt, pred):
    r_gt = {c:i for i,c in enumerate(gt)}
    r_pr = {c:i for i,c in enumerate(pred)}
    common = [c for c in gt if c in r_pr]
    n = len(common)
    total_pairs = n*(n-1)//2
    if total_pairs == 0:
        return 0.0
    concordant = 0
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = common[i], common[j]
            concordant += int((r_gt[ci] < r_gt[cj]) == (r_pr[ci] < r_pr[cj]))
    return float(concordant)/float(total_pairs)

def compute_session_metrics(gt_order, pred_order, vlm_is_correct, started_dt, ended_dt,
                            n_cubes_start=None, n_cubes_end=None):
    gt = normalize_seq(gt_order or [])
    pr = normalize_seq(pred_order or [])

    duration_s = max(0.0, (ended_dt - started_dt).total_seconds())
    prec, rec, f1 = set_precision_recall_f1(gt, pr)
    missing = list(set(gt) - set(pr))
    extra   = list(set(pr) - set(gt))

    ppa  = per_position_accuracy(gt, pr)
    ham  = hamming_distance(gt, pr)
    foot = footrule_distance(gt, pr)
    poa  = kendall_pairwise_accuracy(gt, pr)

    cubes_moved = min(len(gt), len(pr)) - sum(1 for i in range(min(len(gt), len(pr))) if gt[i]==pr[i])
    order_change_ratio = (float(cubes_moved) / max(1, len(gt))) if gt else 0.0

    return {
        "duration_s": round(duration_s, 3),
        "n_cubes_target": len(gt) if gt_order is not None else None,
        "n_cubes_detected_start": (None if n_cubes_start is None else int(n_cubes_start)),
        "n_cubes_detected_end":   (None if n_cubes_end   is None else int(n_cubes_end)),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "missing_colors": missing,
        "extra_colors":   extra,
        "per_position_acc": round(ppa, 4),
        "hamming":          int(ham),
        "spearman_footrule": int(foot),
        "pairwise_acc":     round(poa, 4),
        "cubes_moved": int(cubes_moved),
        "order_change_ratio": round(order_change_ratio, 4),
        "correct_flag": bool(vlm_is_correct),
    }

# ------------------- Helpers de E/S -------------------
def to_dt(rostime):  # rospy.Time -> datetime UTC
    return datetime.datetime.utcfromtimestamp(rostime.to_sec())

def default_base_dir():
    return Path.home() / ".ros" / "rehab_sessions"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_move_or_copy(src: Path, dst: Path, move: bool):
    ensure_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

# ------------------- Servicio principal -------------------
def handle_finalize(req):
    resp = FinalizeSessionResponse(success=False, session_dir="", manifest_json="", error="")
    try:
        # 1) base y carpeta de sesión
        base_dir = Path(req.output_base_dir) if req.output_base_dir else default_base_dir()
        ensure_dir(base_dir)
        session_dir = base_dir / req.session_id
        ensure_dir(session_dir)

        # 2) mover/copiar imágenes
        img1_src = Path(req.image_start_path) if req.image_start_path else None
        img2_src = Path(req.image_end_path)   if req.image_end_path   else None
        img1_dst = session_dir / "first.png"
        img2_dst = session_dir / "second.png"

        if img1_src and img1_src.exists() and img1_src.resolve() != img1_dst.resolve():
            safe_move_or_copy(img1_src, img1_dst, req.move_files)
        if img2_src and img2_src.exists() and img2_src.resolve() != img2_dst.resolve():
            safe_move_or_copy(img2_src, img2_dst, req.move_files)

        # 3) métricas
        started_dt = to_dt(req.started_at)
        ended_dt   = to_dt(req.ended_at)
        metrics = compute_session_metrics(
            gt_order=list(req.gt_order),
            pred_order=list(req.pred_order),
            vlm_is_correct=bool(req.vlm_is_correct),
            started_dt=started_dt,
            ended_dt=ended_dt,
            n_cubes_start=None,
            n_cubes_end=(len(req.pred_order) if req.pred_order else None)
        )

        # 4) manifest
        manifest = {
            "session_id": req.session_id,
            "exercise_type": req.exercise_type,
            "started_at": started_dt.replace(microsecond=0).isoformat() + "Z",
            "ended_at":   ended_dt.replace(microsecond=0).isoformat() + "Z",
            "prompt": req.prompt,
            "camera_topic": req.camera_topic,
            "images": {
                "start": "first.png" if img1_src else "",
                "end":   "second.png" if img2_src else ""
            },
            "ground_truth": {
                "target_order": list(req.gt_order),
                "is_correct": (list(req.gt_order) == list(req.pred_order)) if (req.gt_order and req.pred_order) else None
            },
            "vlm": {
                "model": req.vlm_model,
                "raw_answer": req.vlm_raw_answer,
                "predicted": {
                    "colors_order": list(req.pred_order),
                    "is_correct": bool(req.vlm_is_correct)
                },
                "latency_ms": int(req.vlm_latency_ms),
                "syntax_parse_ok": bool(req.pred_order)
            },
            "metrics": metrics,
            "notes": ""
        }

        # 5) guardar y devolver
        (session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        resp.success = True
        resp.session_dir = str(session_dir)
        resp.manifest_json = json.dumps(manifest, ensure_ascii=False)  # ← para que la GUI lo muestre
        return resp

    except Exception as e:
        resp.error = str(e)
        return resp

def main():
    rospy.init_node("metrics_server", anonymous=False)
    rospy.Service("/vlm/finalize_session", FinalizeSession, handle_finalize)
    rospy.loginfo("metrics_server listo en /vlm/finalize_session (guarda y devuelve manifest)")
    rospy.spin()

if __name__ == "__main__":
    main()
