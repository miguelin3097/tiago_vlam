#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_server.py
Nodo ROS que gestiona el guardado de sesión y el cálculo de métricas.
INTEGRACIÓN COMPLETA:
  - Guarda imágenes y crea carpetas por sesión.
  - Calcula métricas matemáticas de ordenación.
  - Extrae métricas visuales (conteo de cubos) directamente del JSON de Gemini.
"""

import os
import json
import shutil
import datetime
from pathlib import Path

import rospy
from vlm.srv import FinalizeSession, FinalizeSessionResponse


# ==========================================
# 1. UTILIDADES DE NORMALIZACIÓN Y MATEMÁTICAS
# ==========================================

def normalize_color(c):
    """Normaliza nombres de colores (ES/EN) a inglés minúsculas."""
    c = (c or "").strip().lower()
    mapping = {
        "rojo":"red", "verde":"green", "azul":"blue", "amarillo":"yellow",
        "naranja":"orange", "morado":"purple", "violeta":"purple",
        "rosa":"pink", "negro":"black", "blanco":"white",
        "marrón":"brown", "marron":"brown", "cafe":"brown",
        "gris":"gray", "grís":"gray"
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
    """Distancia de edición simple (sustituciones + diferencia longitud)."""
    n = min(len(gt), len(pred))
    base = sum(1 for i in range(n) if gt[i] != pred[i])
    return base + abs(len(gt) - len(pred))

def footrule_distance(gt, pred):
    """Distancia basada en la diferencia de posiciones (ranks)."""
    r_gt = {c:i for i,c in enumerate(gt)}
    r_pr = {c:i for i,c in enumerate(pred)}
    common = r_gt.keys() & r_pr.keys()
    return sum(abs(r_pr[c] - r_gt[c]) for c in common)

def kendall_pairwise_accuracy(gt, pred):
    """Porcentaje de pares relativos correctamente ordenados."""
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
                            extracted_n_start=None, 
                            extracted_n_end=None, 
                            extracted_cubes_moved=None):
    """
    Combina métricas matemáticas (calculadas aquí) con métricas visuales (extraídas de Gemini).
    """
    gt = normalize_seq(gt_order or [])
    pr = normalize_seq(pred_order or [])

    duration_s = max(0.0, (ended_dt - started_dt).total_seconds())
    
    # 1. Métricas de Conjunto (¿Están los colores correctos?)
    prec, rec, f1 = set_precision_recall_f1(gt, pr)
    missing = list(set(gt) - set(pr))
    extra   = list(set(pr) - set(gt))

    # 2. Métricas de Orden (¿Están en la posición correcta?)
    ppa  = per_position_accuracy(gt, pr)
    ham  = hamming_distance(gt, pr)
    foot = footrule_distance(gt, pr)
    poa  = kendall_pairwise_accuracy(gt, pr)

    # 3. Cálculo matemático de cambios (Físico vs Lógico)
    # Cuántos cubos NO coinciden en su posición ideal
    calc_order_mismatches = min(len(gt), len(pr)) - sum(1 for i in range(min(len(gt), len(pr))) if gt[i]==pr[i])
    
    # 4. Fallback para n_end si Gemini no lo reportó
    final_n_end = extracted_n_end if extracted_n_end is not None else len(pr)

    return {
        "duration_s": round(duration_s, 3),
        "n_cubes_target": len(gt) if gt_order is not None else None,
        
        # --- DATOS VISUALES REALES (Extraídos de Gemini) ---
        "n_cubes_detected_start": extracted_n_start,
        "n_cubes_detected_end":   final_n_end,
        "cubes_moved_visual":     extracted_cubes_moved, # Lo que vio Gemini moverse
        
        # --- DATOS MATEMÁTICOS DE ORDEN ---
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "per_position_acc": round(ppa, 4),
        "hamming":          int(ham),
        "spearman_footrule": int(foot),
        "pairwise_acc":     round(poa, 4),
        
        # --- ERRORES ESPECÍFICOS ---
        "missing_colors": missing,
        "extra_colors":   extra,
        "order_mismatches": int(calc_order_mismatches),
        
        "correct_flag": bool(vlm_is_correct),
    }


# ==========================================
# 2. HELPERS DE SISTEMA DE ARCHIVOS
# ==========================================

def to_dt(rostime):
    return datetime.datetime.utcfromtimestamp(rostime.to_sec())

def default_base_dir():
    return Path.home() / ".ros" / "rehab_sessions"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_move_or_copy(src: Path, dst: Path, move: bool):
    ensure_dir(dst.parent)
    try:
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
    except Exception as e:
        rospy.logwarn(f"No se pudo mover/copiar {src} a {dst}: {e}")


# ==========================================
# 3. LÓGICA DEL SERVICIO
# ==========================================

def handle_finalize(req):
    resp = FinalizeSessionResponse(success=False, session_dir="", manifest_json="", error="")
    try:
        # A. Preparar Directorios
        base_dir = Path(req.output_base_dir) if req.output_base_dir else default_base_dir()
        ensure_dir(base_dir)
        
        # Crear carpeta única para la sesión
        session_dir = base_dir / req.session_id
        if not session_dir.exists():
            session_dir.mkdir(parents=True)

        # B. Gestionar Imágenes
        img1_src = Path(req.image_start_path) if req.image_start_path else None
        img2_src = Path(req.image_end_path)   if req.image_end_path   else None
        img1_dst = session_dir / "first.png"
        img2_dst = session_dir / "second.png"

        if img1_src and img1_src.exists() and img1_src.resolve() != img1_dst.resolve():
            safe_move_or_copy(img1_src, img1_dst, req.move_files)
        if img2_src and img2_src.exists() and img2_src.resolve() != img2_dst.resolve():
            safe_move_or_copy(img2_src, img2_dst, req.move_files)

        # C. EXTRACCIÓN DE DATOS VISUALES (Parsing del JSON crudo)
        extracted_n_start = None
        extracted_n_end = None
        extracted_cubes_moved = None
        
        if req.vlm_raw_answer:
            try:
                # Limpieza de Markdown (```json ... ```)
                clean_json = req.vlm_raw_answer.strip()
                if clean_json.startswith("```json"): clean_json = clean_json[7:]
                elif clean_json.startswith("```"): clean_json = clean_json[3:]
                if clean_json.endswith("```"): clean_json = clean_json[:-3]
                clean_json = clean_json.strip()
                
                data = json.loads(clean_json)
                
                # Normalizar claves a minúsculas para búsqueda segura
                data_lower = {k.lower(): v for k, v in data.items()}
                
                if "n_cubes_detected_start" in data_lower:
                    extracted_n_start = int(data_lower["n_cubes_detected_start"])
                
                if "n_cubes_detected_end" in data_lower:
                    extracted_n_end = int(data_lower["n_cubes_detected_end"])
                
                if "cubes_moved" in data_lower:
                    extracted_cubes_moved = int(data_lower["cubes_moved"])
                    
            except Exception as e:
                rospy.logwarn(f"[MetricsServer] Error extrayendo métricas del JSON crudo: {e}")

        # D. Calcular Métricas Finales
        started_dt = to_dt(req.started_at)
        ended_dt   = to_dt(req.ended_at)
        
        metrics = compute_session_metrics(
            gt_order=list(req.gt_order),
            pred_order=list(req.pred_order),
            vlm_is_correct=bool(req.vlm_is_correct),
            started_dt=started_dt,
            ended_dt=ended_dt,
            extracted_n_start=extracted_n_start,
            extracted_n_end=extracted_n_end,
            extracted_cubes_moved=extracted_cubes_moved
        )

        # E. Generar Manifest JSON
        manifest = {
            "session_id": req.session_id,
            "exercise_type": req.exercise_type,
            "timestamps": {
                "started_at": started_dt.replace(microsecond=0).isoformat() + "Z",
                "ended_at":   ended_dt.replace(microsecond=0).isoformat() + "Z"
            },
            "context": {
                "prompt": req.prompt,
                "camera_topic": req.camera_topic
            },
            "images": {
                "start": "first.png" if img1_src else "",
                "end":   "second.png" if img2_src else ""
            },
            "ground_truth": {
                "target_order": list(req.gt_order)
            },
            "vlm_analysis": {
                "model": req.vlm_model,
                "raw_answer": req.vlm_raw_answer, # Guardamos la respuesta original completa
                "predicted": {
                    "colors_order": list(req.pred_order),
                    "is_correct": bool(req.vlm_is_correct)
                },
                "latency_ms": int(req.vlm_latency_ms)
            },
            "metrics": metrics, # Aquí van todas las métricas calculadas
            "notes": ""
        }

        # F. Guardar en Disco y Responder
        manifest_path = session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        
        resp.success = True
        resp.session_dir = str(session_dir)
        resp.manifest_json = json.dumps(manifest, ensure_ascii=False)
        return resp

    except Exception as e:
        rospy.logerr(f"[MetricsServer] Error crítico: {e}")
        resp.error = str(e)
        return resp

def main():
    rospy.init_node("metrics_server", anonymous=False)
    rospy.Service("/vlm/finalize_session", FinalizeSession, handle_finalize)
    rospy.loginfo("metrics_server listo. Guardando sesiones en ~/.ros/rehab_sessions")
    rospy.spin()

if __name__ == "__main__":
    main()