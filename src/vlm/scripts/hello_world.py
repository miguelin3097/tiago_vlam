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
import actionlib  # <--- NUEVO: Necesario para ActionClient
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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
    Extrae orden y assessment del JSON de respuesta.
    """
    if not answer_text:
        return [], None

    txt = answer_text.strip().lower()
    # Intento básico de limpiar markdown si el modelo lo mete
    if txt.startswith("```json"):
        txt = txt.replace("```json", "").replace("```", "")
    
    try:
        answer_json = json.loads(txt)
        return answer_json.get("order", []), answer_json.get("assestment", None)
    except json.JSONDecodeError:
        rospy.logwarn(f"Error decodificando JSON del VLM: {txt[:50]}...")
        return [], None


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
    "Look at the final image and note the order of the cubes from left to right. "
    "The cubes must form a straight horizontal line. "
    "The answer format must be strictly JSON with keys 'order' (list of colors) and "
    "'assestment' (boolean, true if matches GT: [GT]). Return only valid JSON."
)
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
        form.addRow("GT order:", self.gtEdit)

        top = QtWidgets.QHBoxLayout()
        top.addLayout(form, 1)
        top.addWidget(self.btnStart, 0)
        top.addWidget(self.btnReset, 0)
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
        
        # --- Estado para el Robot ---
        self.current_plan = None
        self.ac_pickup = None      # Cliente Acción PickUp
        self.srv_transform = None  # Cliente Servicio Transformación

        # ---------- Señales ----------
        self.btnStart.clicked.connect(self.on_start_clicked)
        self.btnReset.clicked.connect(self.on_reset_clicked)
        self.btnExecute.clicked.connect(self.on_execute_clicked) # Conexión nueva

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
        if not img_path:
            return
        img = cv2.imread(img_path)
        if img is None:
            return

        if not plan_dict:
            self._show_qimage_from_cv(img, self.firstImgLbl)
            self.countdownLbl.setText("Waiting response...")
            self.firstImgLbl.setToolTip("RESET photo")
            QtWidgets.QApplication.processEvents()
            return

        h, w = img.shape[:2]
        
        # Cubes
        cubes = []
        for item in plan_dict.get("cubes", []):
            color_name = normalize_color(str(item.get("color", "")))
            yx = item.get("yx", None)
            if yx is None and "y" in item and "x" in item:
                yx = [item["y"], item["x"]]
            if not isinstance(yx, (list, tuple)) or len(yx) < 2:
                continue
            y_px, x_px = self._denorm_yx1000_to_px(yx[0], yx[1], w, h)
            if y_px is not None:
                cubes.append({"color": color_name, "x": x_px, "y": y_px})

        # Targets
        targets = {}
        targets_raw = plan_dict.get("targets", {})
        # Unificar formato si viene como dict o lista
        it = targets_raw.items() if isinstance(targets_raw, dict) else [(x.get("color",""), x) for x in (targets_raw or [])]
        
        for cname, pos in it:
            c = normalize_color(str(cname))
            yx = pos.get("yx", None)
            if yx is None and "y" in pos and "x" in pos:
                yx = [pos["y"], pos["x"]]
            if not isinstance(yx, (list, tuple)) or len(yx) < 2:
                continue
            y_px, x_px = self._denorm_yx1000_to_px(yx[0], yx[1], w, h)
            if y_px is not None:
                targets[c] = {"x": x_px, "y": y_px}

        # Ajustar targets
        targets = self._fix_or_generate_targets(cubes, targets, plan_dict.get("order", []), w, h)

        # Dibujar
        for item in cubes:
            x, y = int(item["x"]), int(item["y"])
            color_name = item["color"]
            bgr = COLOR_TO_BGR.get(color_name, (0, 255, 255))
            cv2.circle(img, (x, y), 8, bgr, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 10, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img, color_name, (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

        for c, pos in targets.items():
            tx, ty = int(pos["x"]), int(pos["y"])
            bgr = COLOR_TO_BGR.get(c, (255, 255, 255))
            cv2.drawMarker(img, (tx, ty), bgr, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.putText(img, f"target:{c}", (tx+12, ty+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

        # Lineas
        target_points = {c: (int(p["x"]), int(p["y"])) for c, p in targets.items()}
        for item in cubes:
            c = item["color"]
            sx, sy = int(item["x"]), int(item["y"])
            if c in target_points:
                tx, ty = target_points[c]
                bgr = COLOR_TO_BGR.get(c, (200, 200, 200))
                cv2.line(img, (sx, sy), (tx, ty), bgr, 2, cv2.LINE_AA)

        self._show_qimage_from_cv(img, self.secondImgLbl)
        self.secondImgLbl.setToolTip("RESET photo with plan overlay")
        self.countdownLbl.setText("Plan Ready!")

    # =========================================================================
    # ACCIONES DE BOTONES (Slots)
    # =========================================================================

    def on_reset_clicked(self):
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

    def on_execute_clicked(self):
        """Recorre el plan, transforma coordenadas y manda al robot."""
        if not HAS_ROBOT_MODULES:
            self.resultLbl.setText("ERROR: Mensajes del robot no disponibles.")
            return
        if not self.current_plan:
            return

        # Inicializar Clientes (Lazy Loading)
        if self.ac_pickup is None:
            self.ac_pickup = actionlib.SimpleActionClient('/vlm/pickup', PickUpAction)
            rospy.loginfo("Esperando servidor de Acción PickUp...")
            # self.ac_pickup.wait_for_server(rospy.Duration(2.0)) # Opcional timeout

        if self.srv_transform is None:
            rospy.wait_for_service('/vlm/transform_pose')
            self.srv_transform = rospy.ServiceProxy('/vlm/transform_pose', TransformPose)

        # 2. Extraer datos del plan
        cubes_list = self.current_plan.get("cubes", [])
        targets_dict = self.current_plan.get("targets", {})
        order_list = self.current_plan.get("order", [])
        
        # Mapa rápido para buscar cubos por color
        cubes_map = {normalize_color(c.get("color","")): c for c in cubes_list}

        # Dimensiones para des-normalizar
        img_w = 640
        img_h = 480
        if "_image_size" in self.current_plan:
            img_w = self.current_plan["_image_size"].get("width", 640)
            img_h = self.current_plan["_image_size"].get("height", 480)

        self.resultLbl.setText("Executing Sequence (R -> L)...")
        QtWidgets.QApplication.processEvents()

        # --- MODIFICACIÓN: Ordenar de Derecha (Mayor X) a Izquierda (Menor X) ---
        def get_x_coord(color_name):
            c_norm = normalize_color(color_name)
            if c_norm not in cubes_map: return -1
            # yx -> [y, x]. Queremos el índice 1 (x)
            cube_data = cubes_map[c_norm]
            if "yx" in cube_data:
                return cube_data["yx"][1] # Formato VLM 0-1000
            elif "x" in cube_data:
                return cube_data["x"]     # Formato pseudo (pixels)
            return -1

        # Orden descendente (reverse=True)
        sorted_order_list = sorted(order_list, key=get_x_coord, reverse=True)
        rospy.loginfo(f"Execution Order (Right->Left): {sorted_order_list}")

        # 3. Ejecutar Secuencia
        for color_raw in sorted_order_list:
            color = normalize_color(color_raw)
            
            if color not in cubes_map:
                rospy.logwarn(f"Skipping {color}: data missing in plan.")
                continue

            # --- A. OBTENER PIXELES (PICK) ---
            cube_data = cubes_map[color]
            cy_norm, cx_norm = cube_data.get("yx", [0,0])
            py_px, px_px = self._denorm_yx1000_to_px(cy_norm, cx_norm, img_w, img_h)

            #rospy.loginfo(f"Procesando {color}: PickPx({px_px},{py_px}) -> PlacePx({lx_px},{ly_px})")

            # --- C. LLAMAR AL SERVICIO DE TRANSFORMACIÓN (Pixels -> Meters) ---
            try:
                # Transformar PICK
                res_pick = self.srv_transform(cam_x=float(px_px), cam_y=float(py_px), cam_z_depth=1.0) 
                if not res_pick.success: raise Exception("Transform Pick Failed")
                
                # Transformar PLACE
            except Exception as e:
                rospy.logerr(f"Error transformando coordenadas para {color}: {e}")
                continue
            # --- B. PLACE (Destino): Viene de la LISTA FIJA ---
            # Cogemos la coordenada i de nuestra lista hardcodeada
            tgt_x, tgt_y, tgt_z = FIXED_TARGETS_METERS[color]

            rospy.loginfo(f"Moving {color}: Vision({res_pick.base_x:.2f}, {res_pick.base_y:.2f}) -> Fixed({tgt_x}, {tgt_y})")
            # --- D. ENVIAR ACCIÓN AL ROBOT ---
            goal = PickUpGoal()
            goal.pick_x = res_pick.base_x
            goal.pick_y = res_pick.base_y
            #goal.pick_z = res_pick.base_z
            goal.pick_z = 0.65
            goal.place_x = tgt_x
            #goal.place_x = -0.30
            goal.place_y = tgt_y
            #goal.place_y = 0.50
            goal.place_z = tgt_z
            #goal.place_z = 0.50
            goal.use_cube_mode = False
            #goal.torso_height = 0.30
            
            # Opciones
            goal.use_cube_mode = False # Es un cubo, ignoramos Yaw
            # goal.torso_height = 0.30 # ELIMINADO (según tu petición)

            rospy.loginfo(f"Enviando Acción para {color}...")
            self.ac_pickup.send_goal(goal, feedback_cb=self._robot_feedback_cb)
            
            # Esperar a que termine este movimiento antes del siguiente
            while not self.ac_pickup.wait_for_result(rospy.Duration(0.1)):
                QtWidgets.QApplication.processEvents() # Mantiene la GUI viva
            
            result = self.ac_pickup.get_result()
            if not result.success:
                self.resultLbl.setText(f"Error moving {color}: {result.message}")
                # Opcional: return para parar todo, o continue para intentar el siguiente
                continue

        self.resultLbl.setText("Sequence Completed Successfully!")
        self.countdownLbl.setText("Robot Done.")

    def _robot_feedback_cb(self, feedback):
        # Actualizar la GUI con lo que está haciendo el robot
        self.resultLbl.setText(f"Robot: {feedback.current_state}")

    # ----- Flujo Auditoría -----
    def on_start_clicked(self):
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
        pred_order, pred_is_correct = parse_predicted_order(raw)
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