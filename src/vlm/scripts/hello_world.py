#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import datetime
import threading
from pathlib import Path
import cv2


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Servicios (si alguno no está instalado/compilado aún, la GUI lo maneja con try/except)
try:
    from vlm.srv import GeminiAnalyze, GeminiAnalyzeRequest
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# Servicios PlanAction (nuevo)
try:
    from vlm.srv import PlanAction, PlanActionRequest
    HAS_PLAN_ACTION = True
except Exception:
    HAS_PLAN_ACTION = False


from vlm.srv import FinalizeSession, FinalizeSessionRequest

from PyQt5 import QtCore, QtGui, QtWidgets


# =========================
# Utilidades de parsing VLM
# =========================
COLOR_MAP = {
    # es -> canonical
    "rojo": "red", "verde": "green", "azul": "blue", "amarillo": "yellow",
    "naranja": "orange", "morado": "purple", "violeta": "purple", "lila": "purple",
    "rosa": "pink", "negro": "black", "blanco": "white", "marrón": "brown",
    "marron": "brown", "café": "brown", "cafe": "brown", "gris": "gray",
    # en -> canonical
    "red": "red", "green": "green", "blue": "blue", "yellow": "yellow",
    "orange": "orange", "purple": "purple", "violet": "purple", "pink": "pink",
    "black": "black", "white": "white", "brown": "brown", "gray": "gray", "grey": "gray"
}

COLOR_REGEX = re.compile(
    r"\b(" + "|".join(sorted(COLOR_MAP.keys(), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)

def normalize_color(token: str) -> str:
    token = (token or "").strip().lower()
    return COLOR_MAP.get(token, token)

#def parse_predicted_order(answer_text: str):
#    """
#    Extrae una secuencia de colores de la respuesta del VLM.
#    Acepta formatos como:
#      - "Orden: rojo, verde, azul"
#      - "Left to right: red -> green -> blue"
#      - "The order is red, green and blue."
#    Devuelve (lista_colores_normalizados, flag_is_correct)
#    """
#    if not answer_text:
#        return [], None
#
#    txt = answer_text.strip().lower()
#
#    # Detecta 'correcto/incorrecto'
#    is_correct = None
#    if re.search(r"\b(correcto|correcta|es correcto|orden.*correcto)\b", txt):
#        is_correct = True
#    elif re.search(r"\b(incorrecto|incorrecta|no.*correcto|orden.*incorrecto)\b", txt):
#        is_correct = False
#    elif re.search(r"\b(correct|is correct)\b", txt):
#        is_correct = True
#    elif re.search(r"\b(incorrect|not correct)\b", txt):
#        is_correct = False
#
#    # Busca una línea/segmento que parezca enumerar colores
#    # Primero intenta tras palabras clave
#    candidates = []
#    key_patterns = [
#        r"(?:orden|order|left to right|de izquierda a derecha)\s*[:\-]\s*(.*)",
#        r"(?:es|is|are)\s*(.*)"  # fallback
#    ]
#    for pat in key_patterns:
#        m = re.search(pat, txt)
#        if m:
#            candidates.append(m.group(1))
#
#    # si no hay candidatos, usa el texto completo
#    if not candidates:
#        candidates = [txt]
#
#    best_seq = []
#    for segment in candidates:
#        # separadores comunes
#        parts = re.split(r"[,\;\|\>]| y | and ", segment)
#        seq = []
#        for p in parts:
#            for m in COLOR_REGEX.finditer(p):
#                seq.append(normalize_color(m.group(1)))
#        # filtra repetidos consecutivos triviales
#        seq_clean = []
#        for c in seq:
#            if not seq_clean or seq_clean[-1] != c:
#                seq_clean.append(c)
#        if len(seq_clean) > len(best_seq):
#            best_seq = seq_clean
#
#    # como fallback absoluto: saca todos los colores en orden de aparición
#    if not best_seq:
#        seq = [normalize_color(m.group(1)) for m in COLOR_REGEX.finditer(txt)]
#        if seq:
#            best_seq = seq
#
#    return best_seq, is_correct


def parse_predicted_order(answer_text: str):
    """
    Extrae una secuencia de colores de la respuesta del VLM.
    Acepta formatos como:
      - "Orden: rojo, verde, azul"
      - "Left to right: red -> green -> blue"
      - "The order is red, green and blue."
    Devuelve (lista_colores_normalizados, flag_is_correct)
    """
    if not answer_text:
        return [], None

    txt = answer_text.strip().lower()

    answer_json=json.loads(txt)


    return answer_json["order"], answer_json["assestment"]


# =========================
# Buffer de imagen (ROS->CV)
# =========================
class ImageBuffer(object):
    def __init__(self):
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame = None

    def update(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"cv_bridge error: {e}")
            return
        with self._lock:
            self._frame = cv_img

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


# =========================
# GUI principal
# =========================
DEFAULT_TOPIC = "/xtion/rgb/image_raw"
DEFAULT_PROMPT = (
    "Look at the final image and note the order of the cubes from left to right. The cubes must form a straight horizontal line; those not in this line will not count. The answer format must be strictly JSON, where the first key-value pair is the order of the colors from left to right in list form with the key name 'order', and the second key-value pair is your assessment of whether they are correctly ordered, taking into account the GT which is [GT], the key name is 'assestment' and it always will be a boolean value. If you don't know the order of the colors, the answer will be -1. Return only valid JSON, without Markdown, code fences, or explanations — the response must start with { and end with }."
)
DEFAULT_PLAN_PROMPT = (
    "Analyze the image. There are colored cubes on a table. "
    "Return STRICT JSON (no markdown, no comments). Start with '{' and end with '}'. "
    "JSON schema:\n"
    "{\n"
    "  \"cubes\": [{\"color\": \"<color>\", \"yx\": [<int_y_0_1000>, <int_x_0_1000>]}],\n"
    "  \"order\": [\"<color1>\", \"<color2>\", ...],\n"
    "  \"targets\": {\"<color>\": {\"yx\": [<int_y_0_1000>, <int_x_0_1000>] }},\n"
    "  \"image_size\": {\"width\": <int>, \"height\": <int>}\n"
    "}\n"
    "Coordinate system: origin at top-left (x right, y down). "
    "ALL coordinates MUST be normalized integers in the range 0..1000 and returned in [y, x] order inside 'yx'. "
    "Targets represent the DESIRED FINAL POSITIONS and MUST NOT coincide with any current cube position "
    "(keep a minimum Euclidean distance of 40 pixels at 640x480; scale this threshold with the image size). "
    "Arrange targets in a straight horizontal line (same y), left-to-right following 'order', "
    "using reasonable margins (≈8% of width) and equal spacing across the available width. "
    "Colors must be one of: red, green, blue, yellow, orange, purple, pink, black, white, brown, gray."
)

NORM_SCALE = 1000  # el modelo devuelve y,x en 0..1000


# Colores BGR para pintar sobre la imagen
COLOR_TO_BGR = {
    "red":    (0, 0, 255),
    "green":  (0, 255, 0),
    "blue":   (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
    "pink":   (203, 192, 255),
    "black":  (0, 0, 0),
    "white":  (255, 255, 255),
    "brown":  (19, 69, 139),
    "gray":   (128, 128, 128),
}



def iso_utc(dt: datetime.datetime):
    return dt.replace(microsecond=0).isoformat() + "Z"

def new_session_id():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S_session")

class CaptureWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIAGo Rehab — Photo & Session GUI")
        self.resize(1024, 680)

        # ---------- Widgets ----------
        self.topicEdit = QtWidgets.QLineEdit(DEFAULT_TOPIC)
        self.exerciseEdit = QtWidgets.QLineEdit("table_sorting_left_to_right")
        self.gtEdit = QtWidgets.QLineEdit("red, green, blue, yellow")  # GT esperado (coma-separado)
        self.promptEdit = QtWidgets.QPlainTextEdit(DEFAULT_PROMPT)

        self.btnStart  = QtWidgets.QPushButton("Start (5s → photo, 60s → photo)")
        self.btnReset = QtWidgets.QPushButton("Reset (photo → plan)")
        self.planOut = QtWidgets.QPlainTextEdit()
        self.planOut.setReadOnly(True)
        self.planOut.setPlaceholderText("Plan output (JSON)")

        self.countdownLbl = QtWidgets.QLabel("Ready")
        f = self.countdownLbl.font(); f.setPointSize(18); self.countdownLbl.setFont(f)
        self.countdownLbl.setAlignment(QtCore.Qt.AlignCenter)

        self.firstImgLbl  = QtWidgets.QLabel("First photo")
        self.secondImgLbl = QtWidgets.QLabel("Second photo")
        for l in (self.firstImgLbl, self.secondImgLbl):
            l.setFixedSize(480, 360)
            l.setFrameShape(QtWidgets.QFrame.Box)
            l.setAlignment(QtCore.Qt.AlignCenter)

        self.resultLbl = QtWidgets.QLabel("")
        self.resultLbl.setWordWrap(True)
        self.resultLbl.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultLbl.setMinimumHeight(70)

        # ---------- Layout ----------
        form = QtWidgets.QFormLayout()
        form.addRow("Image topic:", self.topicEdit)
        form.addRow("Exercise type:", self.exerciseEdit)
        form.addRow("GT order (comma-separated):", self.gtEdit)

        top = QtWidgets.QHBoxLayout()
        top.addLayout(form, 1)
        top.addWidget(self.btnStart, 0)
        top.addWidget(self.btnReset, 0)


        imgs = QtWidgets.QHBoxLayout()
        imgs.addWidget(self.firstImgLbl, 1)
        imgs.addWidget(self.secondImgLbl, 1)

        promptBox = QtWidgets.QGroupBox("Prompt to VLM")
        vb = QtWidgets.QVBoxLayout(promptBox)
        vb.addWidget(self.promptEdit)

        main = QtWidgets.QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(self.countdownLbl)
        main.addLayout(imgs)
        main.addWidget(promptBox)
        planBox = QtWidgets.QGroupBox("Plan (LLM)")
        vb2 = QtWidgets.QVBoxLayout(planBox)
        vb2.addWidget(self.planOut)
        main.addWidget(planBox)

        main.addWidget(self.resultLbl)

        # ---------- Estado ----------
        self.img_buffer = ImageBuffer()
        self.sub = None

        self.timer_ui = QtCore.QTimer(self); self.timer_ui.setInterval(100)
        self.timer_ui.timeout.connect(self._on_ui_tick)

        self.session_id = None
        self.session_started_at = None
        self.first_done = False
        self.second_done = False
        self.total_secs = 60
        self.countdown_secs = 5

        self.save_tmp_dir = Path.home() / ".ros" / "tiago_photos_tmp"
        self.save_tmp_dir.mkdir(parents=True, exist_ok=True)
        self.first_path = None
        self.second_path = None

        self.vlm_latency_ms = 0
        self.vlm_raw_answer = ""
        self.pred_order = []
        self.pred_is_correct = None

        self.reset_path = None


        # ---------- Señales ----------
        self.btnStart.clicked.connect(self.on_start_clicked)
        self.btnReset.clicked.connect(self.on_reset_clicked)

        # Suscríbete nada más abrir la GUI
        self.start_sub()
        # Re-suscripción si el usuario edita el topic y pulsa Enter o cambia el foco
        #self.topicEdit.editingFinished.connect(self.start_sub)


    # ----- ROS subscribe -----
    def start_sub(self):
        topic = self.topicEdit.text().strip()
        if self.sub:
            self.sub.unregister()
            self.sub = None
        self.sub = rospy.Subscriber(topic, Image, self.img_buffer.update, queue_size=1)
        rospy.loginfo(f"[GUI] Subscribed to {topic}")


    def _capture_reset_photo(self):
        frame = self.img_buffer.get()
        if frame is None:
            rospy.logwarn("[GUI] No frame for RESET photo.")
            self.countdownLbl.setText("No image available! (RESET)")
            return None

        # Guardar a disco (igual que hacía _save_and_show, pero sin mostrar)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fn = self.save_tmp_dir / (f"reset_{ts}.png")
        cv2.imwrite(str(fn), frame)
        self.reset_path = fn

        # Mostrar la imagen en el primer hueco usando la función nueva (sin overlay)
        self._annotate_image_with_plan(str(self.reset_path), {})
        return self.reset_path

    def _call_plan_action(self, image_path: str, prompt: str):
        """
        Llama a /vlm/plan_action y devuelve (raw_plan_json_str, plan_dict).
        plan_dict esperado: {"cubes":[{"color":..., "x":..., "y":...},...], "order":[...]}
        """
        if not HAS_PLAN_ACTION:
            rospy.logwarn("[GUI] PlanAction srv not imported. Skipping call.")
            return "", {}

        try:
            rospy.wait_for_service("/vlm/plan_action", timeout=10.0)
        except Exception as e:
            rospy.logwarn(f"[GUI] /vlm/plan_action not available: {e}")
            return "", {}

        proxy = rospy.ServiceProxy("/vlm/plan_action", PlanAction)
        req = PlanActionRequest(image_path=image_path, prompt=prompt)
        resp = proxy(req)

        if hasattr(resp, "ok") and not resp.ok:
            raise RuntimeError(f"PlanAction error: {getattr(resp, 'error','')}")

        raw = getattr(resp, "plan_json", "") or ""
        try:
            plan = json.loads(raw)
        except Exception as e:
            rospy.logwarn(f"[GUI] Invalid plan_json: {e} | raw: {raw[:200]}")
            plan = {}
        return raw, plan

    def _generate_pseudo_targets(self, plan_dict: dict, img_path: str):
        """
        Genera objetivos pseudoaleatorios deterministas por color dentro del tamaño de la imagen.
        Seed basada en session_id para reproducibilidad.
        Devuelve dict: {"targets":{"red":{"x":..,"y":..}, ...}, "width":W, "height":H}
        """
        import cv2, random
        if not plan_dict or "cubes" not in plan_dict:
            return {}

        img = cv2.imread(img_path)
        if img is None:
            return {}
        h, w = img.shape[:2]

        # Seed determinista con session_id
        seed = (self.session_id or "seed") + "_reset"
        rnd = random.Random(seed)

        # Margen para no caer en los bordes
        margin = int(0.08 * min(w, h))
        targets = {}
        for item in plan_dict.get("cubes", []):
            color = normalize_color(item.get("color", ""))
            tx = rnd.randint(margin, max(margin+1, w - margin))
            ty = rnd.randint(margin, max(margin+1, h - margin))
            targets[color] = {"x": int(tx), "y": int(ty)}

        return {"targets": targets, "width": w, "height": h}

    
    def _show_qimage_from_cv(self, cv_img, target_label):
        #import cv2
        #from PyQt5 import QtGui, QtCore
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qimg)
        target_label.setPixmap(pix.scaled(target_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _denorm_yx1000_to_px(self, y_norm, x_norm, w, h):
        try:
            y = int(round((float(y_norm) / 1000.0) * (h - 1)))
            x = int(round((float(x_norm) / 1000.0) * (w - 1)))
        except Exception:
            return None, None
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return y, x  # devuelve (y_px, x_px)

    def _evenly_spaced_targets(self, order, w, h, y_line=None, margin_ratio=0.08):
        """
        Genera targets equiespaciados sobre una línea horizontal:
        - order: lista de colores en orden izquierda→derecha
        - y_line: si es None, usa ~60% de la altura
        """
        import math
        n = max(1, len(order))
        margin = int(round(margin_ratio * w))
        usable_w = max(1, w - 2 * margin)
        step = usable_w / max(1, n - 1) if n > 1 else 0
        if y_line is None:
            y_line = int(round(0.6 * h))
        targets = {}
        for i, c in enumerate(order):
            x = margin + int(round(i * step))
            y = y_line
            targets[normalize_color(c)] = {"x": x, "y": y}
        return targets

    def _fix_or_generate_targets(self, cubes, targets, order, w, h, min_dist_px=40):
        """
        Si targets falta o alguno está muy cerca de su cube del mismo color,
        recalcula targets equiespaciados en una línea horizontal.
        """
        import math
        if not order:
            order = [c["color"] for c in cubes]

        # Si no hay targets o no cubren todos los colores del orden → generar
        if not targets or any(normalize_color(c) not in targets for c in order):
            return self._evenly_spaced_targets(order, w, h)

        # Comprueba distancia cube↔target color a color
        need_fix = False
        cube_by_color = {normalize_color(c["color"]): (int(c["x"]), int(c["y"])) for c in cubes}
        for c in order:
            key = normalize_color(c)
            if key in cube_by_color and key in targets:
                sx, sy = cube_by_color[key]
                tx, ty = int(targets[key]["x"]), int(targets[key]["y"])
                d = math.hypot(tx - sx, ty - sy)
                if d < min_dist_px:
                    need_fix = True
                    break

        if need_fix:
            return self._evenly_spaced_targets(order, w, h)
        return targets


    def _annotate_image_with_plan(self, img_path: str, plan_dict: dict):
        """
        Si plan_dict está vacío -> muestra la imagen tal cual.
        Si tiene datos -> dibuja:
        - cubes: plan_dict['cubes'] = [{color,x,y}, ...]
        - targets: plan_dict['targets'] = {color: {x,y}, ...}
        - líneas cube->target por color
        Siempre re-muestra en self.firstImgLbl (NO usa _save_and_show).
        """
        #import cv2
        if not img_path:
            return

        img = cv2.imread(img_path)
        if img is None:
            return

        # Si no hay plan -> solo mostrar
        if not plan_dict:
            self._show_qimage_from_cv(img, self.firstImgLbl)
            self.firstImgLbl.setToolTip("RESET photo")
            return

        h, w = img.shape[:2]

        # ----- Con anotaciones -----
        # ---------- CUBES ----------
        cubes = []
        cubes_raw = plan_dict.get("cubes", []) or []
        for item in cubes_raw:
            color_name = normalize_color(str(item.get("color", "")))
            # Esperamos 'yx': [y_norm, x_norm] en 0..1000
            yx = item.get("yx", None)

            # Fallbacks de robustez si el modelo a veces manda otras claves:
            if yx is None and "coords" in item and isinstance(item["coords"], (list, tuple)) and len(item["coords"]) >= 2:
                yx = [item["coords"][0], item["coords"][1]]
            if yx is None and "y" in item and "x" in item:
                # si manda y/x normalizados por separado
                yx = [item["y"], item["x"]]

            if not isinstance(yx, (list, tuple)) or len(yx) < 2:
                continue

            y_px, x_px = self._denorm_yx1000_to_px(yx[0], yx[1], w, h)
            if y_px is None or x_px is None:
                continue
            cubes.append({"color": color_name, "x": x_px, "y": y_px})

        # ---------- TARGETS ----------
        targets = {}
        targets_raw = plan_dict.get("targets", {}) or {}
        if isinstance(targets_raw, dict):
            it = targets_raw.items()
        else:
            # por si alguna vez viene como lista [{color, yx:[..]}]
            it = [(it.get("color",""), it) for it in (targets_raw or [])]

        for cname, pos in it:
            c = normalize_color(str(cname))
            if not isinstance(pos, dict):
                continue
            yx = pos.get("yx", None)
            if yx is None and "coords" in pos and isinstance(pos["coords"], (list, tuple)) and len(pos["coords"]) >= 2:
                yx = [pos["coords"][0], pos["coords"][1]]
            if yx is None and "y" in pos and "x" in pos:
                yx = [pos["y"], pos["x"]]

            if not isinstance(yx, (list, tuple)) or len(yx) < 2:
                continue

            y_px, x_px = self._denorm_yx1000_to_px(yx[0], yx[1], w, h)
            if y_px is None or x_px is None:
                continue
            targets[c] = {"x": x_px, "y": y_px}

        # Asegura que los targets no coinciden con los cubes (o genera equiespaciados si faltan)
        targets = self._fix_or_generate_targets(cubes, targets, plan_dict.get("order", []), w, h, min_dist_px=40)


        # Dibuja cubes
        for item in cubes:
            color_name = normalize_color(str(item.get("color", "")))
            x = int(item.get("x", -1))
            y = int(item.get("y", -1))
            if x < 0 or y < 0:
                continue
            bgr = COLOR_TO_BGR.get(color_name, (0, 255, 255))  # default: amarillo
            cv2.circle(img, (x, y), 8, bgr, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y), 10, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, f"{color_name}", (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

        # Dibuja targets
        for color_name, pos in targets.items():
            c = normalize_color(str(color_name))
            tx = int(pos.get("x", -1))
            ty = int(pos.get("y", -1))
            if tx < 0 or ty < 0:
                continue
            bgr = COLOR_TO_BGR.get(c, (255, 255, 255))  # default: blanco
            marker_type = getattr(cv2, "MARKER_TILTED_CROSS", getattr(cv2, "MARKER_CROSS", 0))
            cv2.drawMarker(img, (tx, ty), bgr, markerType=marker_type,
                        markerSize=20, thickness=2, line_type=cv2.LINE_AA)
            cv2.circle(img, (tx, ty), 10, bgr, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, f"target:{c}", (tx + 12, ty + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

        # Líneas cube->target por color
        target_points = {normalize_color(k): (int(v.get("x", -1)), int(v.get("y", -1)))
                        for k, v in targets.items()
                        if int(v.get("x", -1)) >= 0 and int(v.get("y", -1)) >= 0}

        for item in cubes:
            c = normalize_color(str(item.get("color", "")))
            sx, sy = int(item.get("x", -1)), int(item.get("y", -1))
            if c in target_points and sx >= 0 and sy >= 0:
                tx, ty = target_points[c]
                bgr = COLOR_TO_BGR.get(c, (200, 200, 200))
                cv2.line(img, (sx, sy), (tx, ty), bgr, thickness=2, lineType=cv2.LINE_AA)

        # Leyenda
        #cv2.rectangle(img, (10, 10), (360, 80), (0, 0, 0), thickness=-1)
        #cv2.putText(img, "● cube(center)   x target(center)   line: move-to",
        #            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar en primer hueco (sin _save_and_show)
        self._show_qimage_from_cv(img, self.firstImgLbl)
        self.firstImgLbl.setToolTip("RESET photo with plan overlay")



    def on_reset_clicked(self):
        # Asegura subscripción activa
        #self.start_sub()

        # Toma foto
        path = self._capture_reset_photo()
        if not path:
            self.planOut.setPlainText("ERROR: no image captured for reset.")
            return

        # Prompt (puedes permitir editarlo si quieres; por ahora fijo)
        plan_prompt = DEFAULT_PLAN_PROMPT

        # Llama al LLM
        try:
            raw, plan = self._call_plan_action(str(path), plan_prompt)
        except Exception as e:
            self.planOut.setPlainText(f"ERROR calling plan_action: {e}")
            return

        # Genera objetivos pseudoaleatorios deterministas
        targets_info = self._generate_pseudo_targets(plan, str(path))

        # Adjunta 'targets' si no viniera desde el LLM
        if plan is None:
            plan = {}
        if "targets" not in plan and targets_info:
            plan["targets"] = targets_info.get("targets", {})
            plan["_image_size"] = {"width": targets_info.get("width", 0),
                                "height": targets_info.get("height", 0)}
            plan["_note"] = "targets generated client-side (pseudo-random deterministic by session_id)"

        # Muestra JSON bonito
        try:
            self.planOut.setPlainText(json.dumps(plan, ensure_ascii=False, indent=2))
        except Exception:
            self.planOut.setPlainText(str(plan))
        # Dibuja overlay con cubes/targets sobre la foto RESET
        try:
            self._annotate_image_with_plan(str(self.reset_path), plan)
        except Exception as e:
            rospy.logwarn(f"[GUI] overlay error: {e}")




    # ----- Helpers de imagen -----
    def _save_and_show(self, cv_img, which="first"):
        import cv2
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fn = self.save_tmp_dir / (f"{which}_{ts}.png")
        cv2.imwrite(str(fn), cv_img)
        # Mostrar
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qimg)
        target = self.firstImgLbl if which == "first" else self.secondImgLbl
        target.setPixmap(pix.scaled(target.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        return fn

    # ----- Flujo -----
    def on_start_clicked(self):
        #self.start_sub()
        self.session_id = new_session_id()
        self.session_started_at = datetime.datetime.utcnow()
        self.first_done = False
        self.second_done = False
        self.first_path = None
        self.second_path = None
        self.vlm_latency_ms = 0
        self.vlm_raw_answer = ""
        self.pred_order = []
        self.pred_is_correct = None

        self._t0 = time.time()
        self.timer_ui.start()
        self.btnStart.setEnabled(False)
        self.resultLbl.setText("")
        self.countdownLbl.setText("Countdown: 5")

    def _on_ui_tick(self):
        if self.session_started_at is None:
            return

        elapsed = time.time() - self._t0

        if not self.first_done:
            left = int(max(0, self.countdown_secs - elapsed))
            self.countdownLbl.setText(f"Countdown: {left}")
            if elapsed >= self.countdown_secs:
                self._capture_first()
                self.first_done = True
                self.countdownLbl.setText("First photo taken. Waiting to 60s…")
                return

        if self.first_done and not self.second_done:
            if elapsed >= self.total_secs:
                self._capture_second()
                self.second_done = True
                self.timer_ui.stop()
                self.countdownLbl.setText("Second photo taken. Analyzing…")
                # Continua pipeline: llamar a Gemini y luego a FinalizeSession
                QtCore.QTimer.singleShot(100, self._after_second_photo)

    def _capture_first(self):
        frame = self.img_buffer.get()
        if frame is None:
            rospy.logwarn("[GUI] No frame for FIRST photo.")
            self.countdownLbl.setText("No image available! (FIRST)")
            return
        self.first_path = self._save_and_show(frame, "first")

    def _capture_second(self):
        frame = self.img_buffer.get()
        if frame is None:
            rospy.logwarn("[GUI] No frame for SECOND photo.")
            self.countdownLbl.setText("No image available! (SECOND)")
            return
        self.second_path = self._save_and_show(frame, "second")

    # ----- Integración con servicios -----
    def _call_gemini(self, image1_path: str, image2_path: str, prompt: str):
        """Llama al servicio /gemini_vision/analyze_images y devuelve (raw_answer, pred_order, pred_is_correct, latency_ms)."""
        if not HAS_GEMINI:
            rospy.logwarn("[GUI] Servicio GeminiAnalyze no disponible en tiempo de import. Me salto la llamada.")
            return "", [], None, 0

        try:
            rospy.wait_for_service("/gemini_vision/analyze_images", timeout=10.0)
        except Exception as e:
            rospy.logwarn(f"[GUI] Gemini service not available: {e}")
            return "", [], None, 0

        proxy = rospy.ServiceProxy("/gemini_vision/analyze_images", GeminiAnalyze)
        req = GeminiAnalyzeRequest(image1_path=image1_path, image2_path=image2_path, prompt=prompt)
        t0 = time.time()
        resp = proxy(req)
        t1 = time.time()
        latency_ms = int((t1 - t0) * 1000)

        raw = getattr(resp, "answer", "") if hasattr(resp, "answer") else ""
        # parsea
        pred_order, pred_is_correct = parse_predicted_order(raw)
        return raw, pred_order, pred_is_correct, latency_ms

    def _call_finalize_session(self, **kwargs):
        """Llama al servicio /vlm/finalize_session y devuelve (session_dir, manifest_dict, summary_string)."""
        rospy.wait_for_service("/vlm/finalize_session", timeout=10.0)
        proxy = rospy.ServiceProxy("/vlm/finalize_session", FinalizeSession)
        req = FinalizeSessionRequest()
        req.session_id      = kwargs["session_id"]
        req.exercise_type   = kwargs["exercise_type"]
        req.prompt          = kwargs["prompt"]
        req.camera_topic    = kwargs["camera_topic"]
        req.started_at      = rospy.Time.from_sec(kwargs["started_at_dt"].timestamp())
        req.ended_at        = rospy.Time.from_sec(kwargs["ended_at_dt"].timestamp())
        req.image_start_path = str(kwargs["image_start_path"] or "")
        req.image_end_path   = str(kwargs["image_end_path"] or "")
        req.gt_order        = kwargs["gt_order"] or []
        req.pred_order      = kwargs["pred_order"] or []
        req.vlm_is_correct  = bool(kwargs["pred_is_correct"])
        req.vlm_model       = kwargs["vlm_model"]
        req.vlm_raw_answer  = kwargs["vlm_raw_answer"]
        req.vlm_latency_ms  = int(kwargs["vlm_latency_ms"])
        req.output_base_dir = kwargs.get("output_base_dir","") or ""
        req.move_files      = bool(kwargs.get("move_files", True))

        resp = proxy(req)
        if not resp.success:
            raise RuntimeError("FinalizeSession error: " + resp.error)

        manifest = json.loads(resp.manifest_json) if resp.manifest_json else {}
        # resumen rápido para pantalla
        try:
            acc = manifest["metrics"].get("per_position_acc", 0.0)
            ham = manifest["metrics"].get("hamming", 0)
            cfl = manifest["metrics"].get("correct_flag", False)
            odr = manifest["vlm"]["predicted"].get("colors_order",[])
            if isinstance(odr, str):
                try:
                    odr = json.loads(odr)
                except Exception:
                    odr = [odr]
    
            # convertir la lista a string legible
            odr_str = ", ".join(map(str, odr))
            summary = f"Saved: {resp.session_dir} | Order match: {acc*100:.0f}% | Order Predicted: {odr} | Is Correct?: {cfl}"
        except Exception as e:
            rospy.logwarn(e)
            summary = f"Saved: {resp.session_dir}"
        return resp.session_dir, manifest, summary

    def _after_second_photo(self):
        # 1) Llamar a Gemini
        prompt = self.promptEdit.toPlainText().strip()
        prompt=prompt.replace("[GT]",self.gtEdit.text().strip())
        raw_answer, pred_order, pred_is_correct, latency_ms = self._call_gemini(
            image1_path=str(self.first_path) if self.first_path else "",
            image2_path=str(self.second_path) if self.second_path else "",
            prompt=prompt
        )
        self.vlm_raw_answer = raw_answer
        self.pred_order = pred_order
        self.pred_is_correct = pred_is_correct if pred_is_correct is not None else False
        self.vlm_latency_ms = latency_ms

        # 2) Llamar a FinalizeSession
        gt_list = [c.strip() for c in self.gtEdit.text().split(",") if c.strip()]
        ended_at = datetime.datetime.utcnow()
        try:
            session_dir, manifest, summary = self._call_finalize_session(
                session_id=self.session_id,
                exercise_type=self.exerciseEdit.text().strip(),
                prompt=prompt,
                camera_topic=self.topicEdit.text().strip(),
                started_at_dt=self.session_started_at,
                ended_at_dt=ended_at,
                image_start_path=self.first_path,
                image_end_path=self.second_path,
                gt_order=gt_list,
                pred_order=self.pred_order,
                pred_is_correct=self.pred_is_correct,
                vlm_model="gemini-robotics-er-1.5-preview",
                vlm_raw_answer=self.vlm_raw_answer,
                vlm_latency_ms=self.vlm_latency_ms,
                output_base_dir="",   # vacío -> ~/.ros/rehab_sessions
                move_files=True
            )
            self.resultLbl.setText(summary)
        except Exception as e:
            self.resultLbl.setText(f"ERROR saving session: {e}")

        self.countdownLbl.setText("Done.")
        self.btnStart.setEnabled(True)


# =========================
# main
# =========================
def main():
    import sys
    rospy.init_node("tiago_rehab_gui", anonymous=False, disable_signals=True)

    app = QtWidgets.QApplication(sys.argv)
    w = CaptureWindow()
    w.show()

    # Timer "no-op" para integrar Qt con ROS y permitir Ctrl+C
    t = QtCore.QTimer()
    t.timeout.connect(lambda: None)
    t.start(50)

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
