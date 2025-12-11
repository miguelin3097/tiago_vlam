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
import math
import random

import rospy
import actionlib  # <--- NUEVO: Necesario para ActionClient
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String, Empty

# Servicios VLM (Gemini)
try:
    from vlm.srv import GeminiAnalyze, GeminiAnalyzeRequest
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# Servicios PlanAction (Planificación Visual)
try:
    from vlm.srv import PlanAction, PlanActionRequest
    HAS_PLAN_ACTION = True
except Exception:
    HAS_PLAN_ACTION = False

# Servicios de Sesión
from vlm.srv import FinalizeSession, FinalizeSessionRequest

# --- NUEVO: Servicios y Acciones para el Robot (Transformación y PickUp) ---
try:
    from vlm.srv import TransformPose, TransformPoseRequest
    from vlm.msg import PickUpAction, PickUpGoal
    HAS_ROBOT_MODULES = True
except Exception:
    HAS_ROBOT_MODULES = False
    print("WARNING: No se encontraron los mensajes de vlm (TransformPose o PickUpAction).")

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

def parse_predicted_order(answer_text: str):
    """
    Extrae orden, assessment y feedback del JSON de respuesta.
    Respeta las mayúsculas del feedback.
    """
    if not answer_text:
        return [], None, "No answer received."

    # 1. Limpieza básica (SIN .lower() global)
    txt = answer_text.strip()
    
    # Limpiar markdown de código si Gemini lo incluye
    if txt.startswith("```json"):
        txt = txt[7:]
    elif txt.startswith("```"):
        txt = txt[3:]
    
    if txt.endswith("```"):
        txt = txt[:-3]
    
    txt = txt.strip()
    
    try:
        # 2. Parsear el JSON con el texto original (preservando mayúsculas)
        answer_json = json.loads(txt)
        
        # 3. Normalizar las claves del diccionario a minúsculas
        # Esto es para que funcione igual si Gemini devuelve "Order" o "order"
        data_lower_keys = {k.lower(): v for k, v in answer_json.items()}
        
        # --- A. EL ORDEN (Lo normalizamos nosotros a minúsculas/mayúsculas según prefieras) ---
        raw_order = data_lower_keys.get("order", [])
        # Convertimos los colores a minúsculas para que tu lógica de comparación no falle
        final_order = [str(c).lower() for c in raw_order] 
        
        # --- B. EL ASSESSMENT ---
        # Buscamos 'assessment', 'assestment' (tu typo original) o 'is_correct'
        assessment = data_lower_keys.get("assessment", 
                        data_lower_keys.get("assestment", 
                            data_lower_keys.get("is_correct", None)))

        # --- C. EL FEEDBACK (Lo dejamos INTACTO) ---
        feedback = data_lower_keys.get("feedback", "No specific feedback provided.")
        
        return final_order, assessment, feedback

    except json.JSONDecodeError:
        rospy.logwarn(f"Error decodificando JSON del VLM: {txt[:50]}...")
        # Devolvemos valores seguros en caso de error
        return [], None, "Error processing results."


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
DEFAULT_PROMPT = f"""
        You are a warm, empathetic rehabilitation assistant talking directly to a patient.
        INPUT: Two images (Start vs End).
        
        CRITICAL CONTEXT:
        - Patient sits opposite you.
        - Patient's "Left-to-Right" is your "Right-to-Left".
        
        STRICT FILTERING RULES (Do not ignore this):
        1. **DEFINE THE LINE:** Identify the main group of cubes forming a horizontal line.
        2. **EXCLUDE OUTLIERS:** If a cube is NOT part of that straight line (e.g. it is scattered, far away, or left behind), **DO NOT INCLUDE IT** in the JSON "order" list. The "order" list must ONLY contain the aligned cubes.
        
        TASKS (METRICS & LOGIC):
        1. **COUNT START (Image 1):** Count how many distinct cubes are visible in the first image. -> `n_cubes_detected_start`.
        2. **COUNT END (Image 2):** Count TOTAL distinct cubes in the second image (both in the line AND scattered outliers) -> `n_cubes_detected_end`.
        3 **COUNT MOVEMENT:** Compare Image 1 vs Image 2. How many cubes have changed their position significantly? -> `cubes_moved`.
        4. **CHECK LINE (Image 2):** Do the moved cubes form a straight horizontal line?
        5. **CHECK ORDER (Image 2):** Read aligned cubes RIGHT TO LEFT. Compare with Goal: [GT].

        OUTPUT FORMAT (Strict JSON):
        {{
            "n_cubes_detected_start": (int),
            "n_cubes_detected_end": (int),
            "cubes_moved": (int),
            "order": ["color1", "color2"], 
            "assessment": true/false, 
            "feedback": "TEXT"
        }}
        
        INSTRUCTIONS FOR 'feedback':
        - **TONE:** Supportive.
        - **LENGTH:** Concise (max 2 sentences).
        - **SCENARIO (Cubes excluded):** "I see a cube left out! Please place ALL cubes in the line."
        - **SCENARIO (Perfect):** "Excellent! All cubes are perfectly aligned and ordered."
        - **FORMAT:**
           1. Empathetic message.
           2. Newline (\\n).
        """
DEFAULT_PLAN_PROMPT = (
    "Analyze the image. There are colored cubes on a table. "
    "Return STRICT JSON. Schema: "
    "{'cubes': [{'color': '...', 'yx': [y, x]}], 'order': ['...'], 'targets': {'color': {'yx': [y, x]}}, "
    "'image_size': {'width': w, 'height': h}}. Coordinates 0-1000 normalized."
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

# --- AÑADIR ESTO AL PRINCIPIO (Configuración) ---

# Lista de destinos fijos (en metros, respecto a base_footprint)
# El robot irá rellenando estos huecos en orden.
# Formato: (x, y, z)
# Posicion segun color del cubo
FIXED_TARGETS_METERS = {
    "red": (0.45,  -0.37, 0.66),  # Hueco 1 (Izquierda)
    "blue": (0.58,  -0.42, 0.66),  # Hueco 2 (Centro)
    "yellow": (0.64,  -0.14, 0.66),  # Hueco 3 (Derecha)
    "green": (0.41,  -0.17, 0.66)  # Hueco 4 (Fila de atrás...)
}


def iso_utc(dt: datetime.datetime):
    return dt.replace(microsecond=0).isoformat() + "Z"

def new_session_id():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S_session")

class CaptureWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIAGo Rehab — Photo & Session GUI")
        self.resize(1024, 700)

        # ---------- Widgets ----------
        self.topicEdit = QtWidgets.QLineEdit(DEFAULT_TOPIC)
        self.exerciseEdit = QtWidgets.QLineEdit("table_sorting_left_to_right")
        self.gtEdit = QtWidgets.QLineEdit("red, green, blue, yellow")
        self.promptEdit = QtWidgets.QPlainTextEdit(DEFAULT_PROMPT)

        self.btnStart  = QtWidgets.QPushButton("Start Session (5s → 60s)")
        self.btnReset = QtWidgets.QPushButton("Reset (Plan)")
        
        # --- NUEVO BOTÓN ---
        self.btnExecute = QtWidgets.QPushButton("EXECUTE ON ROBOT (Right->Left)")
        self.btnExecute.setStyleSheet("background-color: #FF9800; font-weight: bold; color: white;")
        self.btnExecute.setEnabled(False) # Se activa solo tras Reset

        self.btnSetTargets = QtWidgets.QPushButton("SET Current as Targets")
        #self.btnSetTargets.setMinimumHeight(50)
        #self.btnSetTargets.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        

        
        
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

        # ---------Publicador
        self.pub_text_command = rospy.Publisher('/web_display/text', String, queue_size=1)
        self.sub_web_cmd = rospy.Subscriber('/web/command', String, self.on_web_command_received)
        self.pub_feedback = rospy.Publisher('/web_display/feedback', String, queue_size=1)
        self.pub_reset_done = rospy.Publisher('/web/reset_done', Empty, queue_size=1)
        self.last_gemini_feedback = "Waiting for analysis..."

        

        # ---------- Layout ----------
        form = QtWidgets.QFormLayout()
        form.addRow("Image topic:", self.topicEdit)
        form.addRow("Exercise type:", self.exerciseEdit)
        form.addRow("GT order:", self.gtEdit)

        top = QtWidgets.QHBoxLayout()
        top.addLayout(form, 1)
        top.addWidget(self.btnStart, 0)
        top.addWidget(self.btnReset, 0)
        top.addWidget(self.btnSetTargets)
        top.addWidget(self.btnExecute, 0) # Añadimos el botón al layout

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
        self.auto_execute_pending = False
        self.fixed_colors = ["red", "blue", "green", "yellow"]
        

        self.saved_targets_visual = {}   # Guardará { "red": [y_norm, x_norm], ... }
        self.saved_targets_physical = {} # Guardará { "red": (x_m, y_m, z_m), ... }
        
        # --- Estado para el Robot ---
        self.current_plan = None
        self.ac_pickup = None      # Cliente Acción PickUp
        self.srv_transform = None  # Cliente Servicio Transformación

        # ---------- Señales ----------
        self.btnStart.clicked.connect(self.on_start_clicked)
        self.btnReset.clicked.connect(self.on_reset_clicked)
        self.btnExecute.clicked.connect(self.on_execute_clicked) # Conexión nueva
        self.btnSetTargets.clicked.connect(self.on_set_targets_clicked)

        self.start_sub()

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

        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fn = self.save_tmp_dir / (f"reset_{ts}.png")
        cv2.imwrite(str(fn), frame)
        self.reset_path = fn

        self._annotate_image_with_plan(str(self.reset_path), {})
        return self.reset_path

    def _call_plan_action(self, image_path: str, prompt: str):
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
        import cv2, random
        if not plan_dict or "cubes" not in plan_dict:
            return {}

        img = cv2.imread(img_path)
        if img is None:
            return {}
        h, w = img.shape[:2]

        seed = (self.session_id or "seed") + "_reset"
        rnd = random.Random(seed)
        margin = int(0.08 * min(w, h))
        targets = {}
        for item in plan_dict.get("cubes", []):
            color = normalize_color(item.get("color", ""))
            tx = rnd.randint(margin, max(margin+1, w - margin))
            ty = rnd.randint(margin, max(margin+1, h - margin))
            targets[color] = {"x": int(tx), "y": int(ty)}

        return {"targets": targets, "width": w, "height": h}

    def _show_qimage_from_cv(self, cv_img, target_label):
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
        return y, x

    def _evenly_spaced_targets(self, order, w, h, y_line=None, margin_ratio=0.08):
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
        import math
        if not order:
            order = [c["color"] for c in cubes]

        if not targets or any(normalize_color(c) not in targets for c in order):
            return self._evenly_spaced_targets(order, w, h)

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
        Dibuja los cubos, los targets (aprendidos o fijos) y las flechas de movimiento.
        Envía el resultado a la GUI y a la pantalla del robot.
        """
        if not img_path:
            return
        img = cv2.imread(img_path)
        if img is None:
            return

        # Si no hay plan, mostramos la imagen limpia esperando
        if not plan_dict:
            self._show_qimage_from_cv(img, self.firstImgLbl)
            self.countdownLbl.setText("Waiting response...")
            self.firstImgLbl.setToolTip("RESET photo")
            # Enviar "Pensando" al robot si quieres, o nada
            QtWidgets.QApplication.processEvents()
            return

        h, w = img.shape[:2]

        DIST_THRESHOLD_PX = 40  # Umbral para considerar que el cubo está "en su sitio"

        # -----------------------------------------------------------
        # 1. OBTENER CUBOS DETECTADOS (SOURCE)
        # -----------------------------------------------------------
        cubes = []
        for item in plan_dict.get("cubes", []):
            color_name = normalize_color(str(item.get("color", "")))
            yx = item.get("yx", None)
            if yx is None and "y" in item and "x" in item:
                yx = [item["y"], item["x"]]
            
            if not isinstance(yx, (list, tuple)) or len(yx) < 2:
                continue
            
            # Convertir 0-1000 -> Píxeles
            y_px, x_px = self._denorm_yx1000_to_px(yx[0], yx[1], w, h)
            if y_px is not None:
                cubes.append({"color": color_name, "x": x_px, "y": y_px})

        # Lista rápida de colores disponibles para filtrar targets
        available_colors = {c["color"] for c in cubes}

        # -----------------------------------------------------------
        # 2. DETERMINAR TARGETS (DESTINATION)
        # -----------------------------------------------------------
        targets = {}

        # LÓGICA DE PRIORIDAD:
        # A. Si el usuario guardó targets con el botón "SET Targets", usamos esos.
        # B. Si no, usamos los valores Hardcodeados por defecto.
        
        if self.saved_targets_visual:
            source_targets_config = self.saved_targets_visual
        else:
            # Defaults por si no has hecho "teaching" todavía
            source_targets_config = {
                "yellow": [480, 497], # [Y, X]
                "green":  [790, 479],
                "blue":   [580, 804],
                "red":    [771, 771],
                # Añade más defaults si quieres
            }

        # Procesar y filtrar targets
        for color, yx in source_targets_config.items():
            # Solo creamos target si existe un cubo de ese color
            if color in available_colors:
                y_norm, x_norm = yx[0], yx[1]
                y_px, x_px = self._denorm_yx1000_to_px(y_norm, x_norm, w, h)
                
                if y_px is not None and x_px is not None:
                    targets[color] = {"x": x_px, "y": y_px}

        # -----------------------------------------------------------
        # 3. DIBUJAR Y CALCULAR MOVIMIENTOS
        # -----------------------------------------------------------
        active_moves = [] # Guardaremos aquí las flechas que hay que pintar

        # A) Dibujar Cubos y chequear distancias
        for item in cubes:
            x, y = int(item["x"]), int(item["y"])
            c = item["color"]
            bgr = COLOR_TO_BGR.get(c, (0, 255, 255))
            
            # Círculo del cubo
            cv2.circle(img, (x, y), 8, bgr, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 10, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img, c, (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

            # Verificar distancia al target (si tiene uno asignado)
            if c in targets:
                t_pos = targets[c]
                tx, ty = int(t_pos["x"]), int(t_pos["y"])
                
                dist = math.hypot(tx - x, ty - y)
                
                if dist < DIST_THRESHOLD_PX:
                    # YA ESTÁ EN SU SITIO -> Pintar "OK"
                    cv2.putText(img, "OK", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # NECESITA MOVERSE -> Guardar para pintar flecha luego
                    active_moves.append({
                        "color": c,
                        "start": (x, y),
                        "end": (tx, ty)
                    })

        # B) Dibujar Targets (Cruces)
        for c, pos in targets.items():
            tx, ty = int(pos["x"]), int(pos["y"])
            bgr = COLOR_TO_BGR.get(c, (255, 255, 255))
            # Cruz grande
            cv2.drawMarker(img, (tx, ty), bgr, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            # Etiqueta
            cv2.putText(img, f"T:{c}", (tx+10, ty+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

        # C) Dibujar Flechas de movimiento (Solo para los activos)
        for move in active_moves:
            sx, sy = move["start"]
            tx, ty = move["end"]
            color_name = move["color"]
            bgr = COLOR_TO_BGR.get(color_name, (200, 200, 200))
            
            # Flecha desde el cubo hasta el target
            cv2.arrowedLine(img, (sx, sy), (tx, ty), bgr, 3, cv2.LINE_AA, tipLength=0.05)

        # -----------------------------------------------------------
        # 4. SALIDA
        # -----------------------------------------------------------
        
        # A) Mostrar en la GUI del PC
        self._show_qimage_from_cv(img, self.secondImgLbl)
        self.secondImgLbl.setToolTip("Plan Visualization")
        self.countdownLbl.setText("Plan Ready!")

    # =========================================================================
    # ACCIONES DE BOTONES (Slots)
    # =========================================================================

    def on_reset_clicked(self):

        self.btnExecute.setEnabled(False)
        path = self._capture_reset_photo()
        if not path:
            self.planOut.setPlainText("ERROR: no image captured for reset.")
            return

        # Llamada al Planificador (VLM)
        try:
            raw, plan = self._call_plan_action(str(path), DEFAULT_PLAN_PROMPT)
        except Exception as e:
            self.planOut.setPlainText(f"ERROR calling plan_action: {e}")
            return
        
        # Generar targets si no vinieron
        targets_info = self._generate_pseudo_targets(plan, str(path))
        if plan is None: plan = {}
        if "targets" not in plan and targets_info:
            plan["targets"] = targets_info.get("targets", {})
            plan["_image_size"] = {"width": targets_info.get("width",0), "height": targets_info.get("height",0)}

        # 1. GUARDAR PLAN Y ACTIVAR EJECUCIÓN
        self.current_plan = plan
        self.btnExecute.setEnabled(True)
        self.countdownLbl.setText("Plan Ready. Press EXECUTE to move robot.")

        

        # Visualizar
        try:
            self.planOut.setPlainText(json.dumps(plan, ensure_ascii=False, indent=2))
        except Exception:
            self.planOut.setPlainText(str(plan))
        
        try:
            self._annotate_image_with_plan(str(self.reset_path), plan)
        except Exception as e:
            rospy.logwarn(f"[GUI] overlay error: {e}")


        # --- CÓDIGO NUEVO: EL GATILLO AUTOMÁTICO ---
        if self.auto_execute_pending:
            rospy.loginfo("Bandera detectada: Ejecutando movimiento automáticamente.")
            
            # 1. Bajamos la bandera (para que no se ejecute solo la próxima vez)
            self.auto_execute_pending = False
            
            # 2. Pulsamos EJECUTAR directamente
            QtCore.QMetaObject.invokeMethod(self.btnExecute, "click")

    def on_set_targets_clicked(self):
        """Guarda la posición actual de los cubos como el nuevo OBJETIVO."""
        if not self.current_plan or "cubes" not in self.current_plan:
            self.resultLbl.setText("No cubes detected yet. Analyze first!")
            return

        # Inicializar servicio de transformación si no existe
        if HAS_ROBOT_MODULES and self.srv_transform is None:
            rospy.wait_for_service('/vlm/transform_pose')
            self.srv_transform = rospy.ServiceProxy('/vlm/transform_pose', TransformPose)

        # Limpiamos targets anteriores
        self.saved_targets_visual = {}
        self.saved_targets_physical = {}

        img_w = self.current_plan.get("_image_size", {}).get("width", 640)
        img_h = self.current_plan.get("_image_size", {}).get("height", 480)

        cubes = self.current_plan.get("cubes", [])
        count = 0

        for item in cubes:
            color = normalize_color(item.get("color", ""))
            if not color: continue

            # 1. GUARDAR VISUAL (Normalizado 0-1000)
            yx = item.get("yx", None) # [y, x]
            if yx:
                self.saved_targets_visual[color] = yx
                
                # 2. CALCULAR Y GUARDAR FÍSICO (Metros)
                # Necesitamos convertir a pixeles para pedir la transformación
                y_px, x_px = self._denorm_yx1000_to_px(yx[0], yx[1], img_w, img_h)
                
                if HAS_ROBOT_MODULES and y_px is not None:
                    try:
                        # Preguntamos al robot: "¿Dónde es este píxel en el mundo real?"
                        res = self.srv_transform(cam_x=float(x_px), cam_y=float(y_px), cam_z_depth=1.0)
                        if res.success:
                            # Guardamos la posición base_x, base_y y la altura z (podemos forzar z=0.85 si es mesa)
                            self.saved_targets_physical[color] = (res.base_x, res.base_y, 0.85)
                            count += 1
                    except Exception as e:
                        rospy.logerr(f"Error transforming target for {color}: {e}")

        # Feedback visual
        msg = f"Saved {count} targets! (Redrawing...)"
        self.resultLbl.setText(msg)
        print(f"VISUAL TARGETS: {self.saved_targets_visual}")
        print(f"PHYSICAL TARGETS: {self.saved_targets_physical}")

        # Redibujar la imagen para que aparezcan las cruces en los nuevos sitios
        if hasattr(self, 'last_img_path'): # Asegúrate de guardar el path de la imagen en algún lado
            self._annotate_image_with_plan(self.first_path, self.current_plan) # first_path suele ser la guardada
    
    def on_web_command_received(self, msg):
        """
        Callback que se ejecuta cuando llega un mensaje al topic /web/command
        """
        command = msg.data.lower().strip()
        
        if command == "start":
            rospy.loginfo("CMD WEB: START -> BARAJANDO SECUENCIA")
            
            # 1. Barajar colores
            new_sequence = self.fixed_colors.copy()
            random.shuffle(new_sequence)
            
            # 2. Guardar internamente
            self.gt_order = new_sequence
            sequence_str = ", ".join(new_sequence)
            
            # 3. Enviar a la pantalla del robot (Para el paciente)
            self.pub_text_command.publish(sequence_str)
            
            # 4. ACTUALIZAR GUI DEL PC (Para ti)
            # Actualizamos la caja de texto donde se ve el Ground Truth
            self.gtEdit.setText(sequence_str)
            
            # 5. Iniciar la captura
            if self.btnStart.isEnabled():
                QtCore.QMetaObject.invokeMethod(self.btnStart, "click")
                
        elif command == "stop":
            rospy.loginfo("Recibido comando STOP remoto desde la Web.")
            # Solo hacemos caso si ya hemos empezado (primera foto hecha) pero no acabado
            if self.first_done and not self.second_done:
                # Usamos QMetaObject para llamar a la función en el hilo principal de la GUI
                QtCore.QMetaObject.invokeMethod(self, "finish_exercise", QtCore.Qt.QueuedConnection)
        elif command == "reset":
            rospy.loginfo("CMD WEB: RESET -> PREPARING AUTO-SEQUENCE")
            
            # 1. Activamos la bandera: "Cuando tengas el plan, ejecútalo"
            self.auto_execute_pending = True
            
            # 2. Pulsamos el botón que pide el plan a Gemini
            if self.btnReset.isEnabled():
                QtCore.QMetaObject.invokeMethod(self.btnReset, "click")

    @QtCore.pyqtSlot()  # Decorador recomendado para ser llamado desde invokeMethod
    def finish_exercise(self):
        """
        Se llama manualmente al pulsar STOP en la web.
        Toma la segunda foto y lanza el análisis.
        """
        rospy.loginfo("Finalizando ejercicio por orden manual...")
        
        # 1. Parar el timer de la interfaz
        self.timer_ui.stop()
        
        # 2. Tomar la segunda foto
        self._capture_second()
        self.second_done = True
        
        # 3. Actualizar GUI
        self.countdownLbl.setText("Second photo taken. Analyzing…")
        
        # 4. Lanzar el análisis (con un pequeño retardo para que la GUI respire)
        QtCore.QTimer.singleShot(100, self._after_second_photo)

    def on_execute_clicked(self):
        """
        Ejecuta la secuencia de pick & place para TODOS los cubos detectados.
        - Ignora la lista 'order' del VLM (para evitar olvidos).
        - Ordena físicamente de Derecha a Izquierda (seguridad).
        - Filtra movimientos si el cubo ya está cerca del target (<40px).
        """
        if not HAS_ROBOT_MODULES:
            self.resultLbl.setText("ERROR: Mensajes del robot no disponibles.")
            return
        if not self.current_plan:
            return

        # -------------------------------------------------------------------------
        # 1. INICIALIZACIÓN DE CLIENTES (Lazy Loading)
        # -------------------------------------------------------------------------
        if self.ac_pickup is None:
            self.ac_pickup = actionlib.SimpleActionClient('/vlm/pickup', PickUpAction)
            rospy.loginfo("Conectando con servidor PickUp...")
            # self.ac_pickup.wait_for_server(rospy.Duration(2.0))

        if self.srv_transform is None:
            rospy.wait_for_service('/vlm/transform_pose')
            self.srv_transform = rospy.ServiceProxy('/vlm/transform_pose', TransformPose)

        # -------------------------------------------------------------------------
        # 2. CONFIGURACIÓN
        # -------------------------------------------------------------------------
        
        # Targets Visuales (0-1000) - Deben coincidir con los que pintas en pantalla
        if self.saved_targets_visual:
            TARGETS_NORM_YX = self.saved_targets_visual
        else:
            TARGETS_NORM_YX = {
                "yellow": [480, 497], 
                "green":  [790, 479],
                "blue":   [580, 804],
                "red":    [771, 771],
            }

        DIST_THRESHOLD_PX = 40  # Si dist < 40px, consideramos que ya está puesto

        # Dimensiones de la imagen para des-normalizar
        img_w = 640
        img_h = 480
        if "_image_size" in self.current_plan:
            img_w = self.current_plan["_image_size"].get("width", 640)
            img_h = self.current_plan["_image_size"].get("height", 480)

        self.resultLbl.setText("Executing ALL detected cubes (Right -> Left)...")
        QtWidgets.QApplication.processEvents()

        # -------------------------------------------------------------------------
        # 3. PREPARACIÓN DE LA LISTA DE EJECUCIÓN
        # -------------------------------------------------------------------------
        
        cubes_list = self.current_plan.get("cubes", [])
        
        # Mapa rápido: "color" -> {datos del cubo}
        cubes_map = {normalize_color(c.get("color","")): c for c in cubes_list}
        
        # Lista de colores encontrados (ignoramos 'order' del VLM, usamos lo que vemos)
        found_colors = [normalize_color(c.get("color","")) for c in cubes_list if c.get("color")]

        # Función helper para obtener coordenada X (para ordenar)
        def get_x_coord(color_name):
            c_norm = normalize_color(color_name)
            if c_norm not in cubes_map: return -1
            cube_data = cubes_map[c_norm]
            if "yx" in cube_data:
                return cube_data["yx"][1] # yx[1] es X
            elif "x" in cube_data:
                return cube_data["x"]
            return -1

        # ORDENAR: De mayor X (derecha) a menor X (izquierda) para evitar colisiones
        sorted_execution_list = sorted(found_colors, key=get_x_coord, reverse=True)
        rospy.loginfo(f"Execution Order (Auto-detected R->L): {sorted_execution_list}")

        # -------------------------------------------------------------------------
        # 4. BUCLE DE EJECUCIÓN
        # -------------------------------------------------------------------------
        for color in sorted_execution_list:
            
            # --- A. VERIFICAR DATOS ---
            if color not in cubes_map:
                continue

            cube_data = cubes_map[color]
            cy_norm, cx_norm = cube_data.get("yx", [0,0])
            
            # Convertir posición del cubo a Píxeles
            py_px, px_px = self._denorm_yx1000_to_px(cy_norm, cx_norm, img_w, img_h)

            if py_px is None or px_px is None:
                rospy.logwarn(f"Skipping {color}: Invalid pixel coordinates.")
                continue

            # --- B. FILTRO DE DISTANCIA (¿Ya está en su sitio?) ---
            should_move = True
            
            if color in TARGETS_NORM_YX:
                ty_norm, tx_norm = TARGETS_NORM_YX[color]
                ty_px, tx_px = self._denorm_yx1000_to_px(ty_norm, tx_norm, img_w, img_h)
                
                if ty_px is not None and tx_px is not None:
                    # Distancia Euclidiana
                    dist = math.hypot(px_px - tx_px, py_px - ty_px)
                    
                    if dist < DIST_THRESHOLD_PX:
                        msg = f"SKIPPING {color}: Already at target (Dist: {dist:.1f}px)"
                        rospy.loginfo(msg)
                        self.resultLbl.setText(msg)
                        QtWidgets.QApplication.processEvents()
                        should_move = False
                    else:
                        msg = f"MOVING{color}: Target (Dist: {dist:.1f}px)"
                        rospy.loginfo(msg)
                        self.resultLbl.setText(msg)
            
            if not should_move:
                continue

            # --- C. VERIFICAR DESTINO FÍSICO ---
            if color not in FIXED_TARGETS_METERS:
                rospy.logwarn(f"Skipping {color}: No physical target in FIXED_TARGETS_METERS.")
                continue

            # --- D. TRANSFORMACIÓN (2D -> 3D) ---
            try:
                # Asumimos profundidad Z=1.0m o lo que use tu cámara cenital
                res_pick = self.srv_transform(cam_x=float(px_px), cam_y=float(py_px), cam_z_depth=1.0) 
                if not res_pick.success: 
                    raise Exception("Transform Pick Failed")
            except Exception as e:
                rospy.logerr(f"Error transformando coordenadas para {color}: {e}")
                continue

            tgt_x, tgt_y, tgt_z = FIXED_TARGETS_METERS[color]
            rospy.loginfo(f"Moving {color} -> Fixed({tgt_x}, {tgt_y})")

            # --- E. ENVIAR ACCIÓN AL ROBOT ---
            goal = PickUpGoal()
            goal.pick_x = res_pick.base_x
            goal.pick_y = res_pick.base_y
            goal.pick_z = 0.65 # Altura segura para agarrar (mesa ~0.8m?)
            
            goal.place_x = tgt_x
            goal.place_y = tgt_y
            goal.place_z = tgt_z
            
            goal.use_cube_mode = False # Asumimos modo normal

            self.ac_pickup.send_goal(goal, feedback_cb=self._robot_feedback_cb)
            
            # Esperar a que termine este movimiento antes del siguiente
            while not self.ac_pickup.wait_for_result(rospy.Duration(0.1)):
                QtWidgets.QApplication.processEvents()
            
            result = self.ac_pickup.get_result()
            if not result.success:
                self.resultLbl.setText(f"Error moving {color}: {result.message}")
                # Decisión: Si falla uno, continuamos con el siguiente
                continue

        # -------------------------------------------------------------------------
        # 5. FINALIZACIÓN
        # -------------------------------------------------------------------------

        rospy.loginfo("Movimiento finalizado. Avisando a pantalla web.")
        self.pub_reset_done.publish(Empty())
        self.resultLbl.setText("Sequence Completed!")
        self.countdownLbl.setText("Robot Done.")

    def _robot_feedback_cb(self, feedback):
        # Actualizar la GUI con lo que está haciendo el robot
        self.resultLbl.setText(f"Robot: {feedback.current_state}")

    # ----- Flujo Auditoría -----
    def on_start_clicked(self):

        # --- NUEVO: ENVIAR TEXTO A LA PANTALLA ---
        # Leemos lo que hay en el cuadro 'GT order'
        raw_text = self.gtEdit.text()
        
        if raw_text:
            # 1. Traducir/Limpiar (usando tu COLOR_MAP global)
            # Entrada: "rojo, verde" -> Salida: "red, green"
            parts = []
            for p in raw_text.split(','):
                p = p.strip().lower()
                if p in COLOR_MAP:
                    parts.append(COLOR_MAP[p])
                else:
                    parts.append(p)
            
            final_text = ", ".join(parts)
            
            # 2. Publicar
            rospy.loginfo(f"Enviando GT a pantalla: {final_text}")
            self.pub_text_command.publish(final_text)
        else:
            # Si está vacío, podemos borrar la pantalla o mandar aviso
            self.pub_text_command.publish("ESPERANDO...")

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

        # PARTE 1: Cuenta atrás inicial (5 segundos) -> Sigue IGUAL
        if not self.first_done:
            left = int(max(0, self.countdown_secs - elapsed))
            self.countdownLbl.setText(f"Countdown: {left}")
            if elapsed >= self.countdown_secs:
                self._capture_first()
                self.first_done = True
                # CAMBIO: Mensaje indicando espera manual
                self.countdownLbl.setText("Exercise running... Press STOP on Web to finish.")
                return

        # PARTE 2: Espera indefinida -> CAMBIO RADICAL
        if self.first_done and not self.second_done:
            # ANTES: Comprobaba if elapsed >= self.total_secs
            # AHORA: No hacemos nada, solo actualizamos el tiempo transcurrido visualmente
            
            # Mostramos cuánto tiempo lleva el paciente
            current_exercise_time = int(elapsed - self.countdown_secs)
            self.countdownLbl.setText(f"Exercise Time: {current_exercise_time}s (Waiting for STOP)")
            
            # ELIMINADO: self._capture_second() automático
            pass

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

    def _call_gemini(self, image1_path: str, image2_path: str, prompt: str):
        if not HAS_GEMINI:
            return "", [], None, 0
        try:
            rospy.wait_for_service("/gemini_vision/analyze_images", timeout=10.0)
        except Exception:
            return "", [], None, 0
        proxy = rospy.ServiceProxy("/gemini_vision/analyze_images", GeminiAnalyze)
        req = GeminiAnalyzeRequest(image1_path=image1_path, image2_path=image2_path, prompt=prompt)
        t0 = time.time()
        resp = proxy(req)
        t1 = time.time()
        latency_ms = int((t1 - t0) * 1000)
        raw = getattr(resp, "answer", "") if hasattr(resp, "answer") else ""
        pred_order, pred_is_correct, feedback = parse_predicted_order(raw)
        self.last_gemini_feedback = feedback
        return raw, pred_order, pred_is_correct, latency_ms

    def _call_finalize_session(self, **kwargs):
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
        try:
            acc = manifest["metrics"].get("per_position_acc", 0.0)
            cfl = manifest["metrics"].get("correct_flag", False)
            odr = manifest["vlm"]["predicted"].get("colors_order",[])
            summary = f"Saved: {resp.session_dir} | Match: {acc*100:.0f}% | Correct?: {cfl}"
        except Exception:
            summary = f"Saved: {resp.session_dir}"
        return resp.session_dir, manifest, summary

    def _after_second_photo(self):
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
                output_base_dir="", 
                move_files=True
            )
            self.resultLbl.setText(summary)
            rospy.loginfo(f"Enviando feedback a pantalla robot...")
            self.pub_feedback.publish(self.last_gemini_feedback)
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

    t = QtCore.QTimer()
    t.timeout.connect(lambda: None)
    t.start(50)

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()