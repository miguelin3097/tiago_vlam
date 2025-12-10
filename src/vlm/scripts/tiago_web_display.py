#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import threading
import time
import socket
import numpy as np
import json
from flask import Flask, Response, jsonify
from sensor_msgs.msg import Image
from std_msgs.msg import String, Empty
from cv_bridge import CvBridge, CvBridgeError

# ==========================================
# CONFIGURACIÓN
# ==========================================
TOPIC_IMG = '/robot/xdisplay'
TOPIC_TEXT = '/web_display/text'
TOPIC_CMD = '/web/command'
TOPIC_FEEDBACK = '/web_display/feedback'
TOPIC_RESET_DONE = '/web/reset_done'
WEB_PORT = 5000

# COLORES
COLOR_DB = {
    "RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0),
    "YELLOW": (0, 255, 255), "ORANGE": (0, 165, 255), "PURPLE": (128, 0, 128),
    "PINK": (147, 20, 255), "BLACK": (50, 50, 50), "WHITE": (255, 255, 255)
}

# ESTADO GLOBAL
app = Flask(__name__)
latest_image = None
image_lock = threading.Lock()
pub_command = None

# VARIABLES DE CONTROL
doing_countdown = False
pending_text = None 
current_difficulty = "TEXT" 
last_received_text = ""
feedback_ready = False 
doing_reset_move = False # Flag para saber si el robot se mueve

# ==========================================
# CLASE ROS
# ==========================================
class WebDisplayNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.sub_img = rospy.Subscriber(TOPIC_IMG, Image, self.image_callback)
        self.sub_text = rospy.Subscriber(TOPIC_TEXT, String, self.text_callback)
        self.sub_feedback = rospy.Subscriber(TOPIC_FEEDBACK, String, self.feedback_callback)
        self.sub_reset_done = rospy.Subscriber(TOPIC_RESET_DONE, Empty, self.reset_done_callback)
        
        global pub_command
        pub_command = rospy.Publisher(TOPIC_CMD, String, queue_size=1)

    def image_callback(self, msg):
        global latest_image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with image_lock: latest_image = cv_img
        except CvBridgeError as e: rospy.logerr(f"Error bridge: {e}")

    def text_callback(self, msg):
        global pending_text, doing_countdown, last_received_text
        last_received_text = msg.data 
        if doing_countdown: pending_text = msg.data
        else: update_image_with_text(msg.data)

    def feedback_callback(self, msg):
        global feedback_ready
        rospy.loginfo(f"Feedback recibido: {msg.data}")
        show_results_screen(msg.data)
        feedback_ready = True

    def reset_done_callback(self, msg):
        """Se llama cuando el robot ha terminado físicamente el reset."""
        global doing_reset_move
        rospy.loginfo("Reset físico completado. Volviendo a instrucciones.")
        
        # 1. Mostrar instrucciones
        show_instructions()
        
        # 2. IMPORTANTE: Bajar la bandera para que el JS sepa que hemos acabado
        doing_reset_move = False 

# ==========================================
# UTILIDADES GRÁFICAS
# ==========================================
def create_image_canvas(bg_color=(0, 0, 0)):
    h, w = 600, 1024
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bg_color 
    return img, w, h

def draw_centered_text(img, text, y, font, scale, color, thickness):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

# --- PANTALLA DE RESULTADOS ---
def show_results_screen(text_report):
    global latest_image
    img, w, h = create_image_canvas(bg_color=(50, 20, 0)) 
    font = cv2.FONT_HERSHEY_SIMPLEX

    draw_centered_text(img, "RESULTS", 60, font, 1.5, (255, 255, 255), 3)
    cv2.line(img, (50, 80), (w-50, 80), (200, 200, 200), 2)

    explicit_lines = text_report.split('\n')
    final_lines = []
    
    for explicit_line in explicit_lines:
        clean_line = explicit_line.strip()
        if len(clean_line) > 0:
            clean_line = clean_line[0].upper() + clean_line[1:]
        
        words = clean_line.split(' ')
        current_line = ""
        for word in words:
            if len(current_line) + len(word) < 45:
                current_line += word + " "
            else:
                final_lines.append(current_line)
                current_line = word + " "
        final_lines.append(current_line)

    start_y = 140
    line_height = 50
    
    for i, line in enumerate(final_lines[:8]):
        color = (180, 255, 180) 
        if "Line:" in line or "Order:" in line or "Cubes" in line:
            color = (255, 255, 0)
        draw_centered_text(img, line, start_y + (i * line_height), font, 1.1, color, 2)

    draw_centered_text(img, "Press RESET to start new session", h-30, font, 0.7, (100, 100, 255), 1)
    with image_lock: latest_image = img

def show_instructions():
    global latest_image
    img, w, h = create_image_canvas(bg_color=(20, 20, 20))
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_centered_text(img, "EXERCISE GOAL:", 80, font, 1.0, (180, 180, 180), 2)
    draw_centered_text(img, "SORT BY COLOR", 200, font, 2.5, (255, 255, 255), 5)
    draw_centered_text(img, "Place cubes in a", 300, font, 1.5, (200, 200, 200), 2)
    draw_centered_text(img, "HORIZONTAL LINE", 370, font, 2.0, (0, 255, 255), 4)
    draw_centered_text(img, "( Left  --->  Right )", 450, font, 1.2, (200, 200, 200), 2)
    draw_centered_text(img, "Select difficulty & Press START", 560, font, 0.8, (0, 200, 0), 2)
    with image_lock: latest_image = img

def update_image_with_text(text, bg_color=(0, 0, 0), title="SEQUENCE:"):
    global latest_image, current_difficulty
    img, w, h = create_image_canvas(bg_color)
    if "," in text: parts = [p.strip().upper() for p in text.split(',')]
    else: parts = [text.strip().upper()]
    font = cv2.FONT_HERSHEY_SIMPLEX
    if title: cv2.putText(img, title, (50, 80), font, 1, (200, 200, 200), 2)

    if current_difficulty == "TEXT":
        formatted_text = " -> ".join(parts)
        font_scale = 1.8; thickness = 4
        (text_w, text_h), _ = cv2.getTextSize(formatted_text, font, font_scale, thickness)
        if text_w > (w - 60): 
            font_scale = 1.0; thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(formatted_text, font, font_scale, thickness)
        x = (w - text_w) // 2; y = (h + text_h) // 2
        cv2.putText(img, formatted_text, (x, y), font, font_scale, (255, 255, 255), thickness)
        cv2.line(img, (x, y + 25), (x + text_w, y + 25), (0, 255, 255), 2)
    else:
        num_cubes = len(parts); cube_size = 120; gap = 80
        total_width = (num_cubes * cube_size) + ((num_cubes - 1) * gap)
        start_x = (w - total_width) // 2; center_y = h // 2; top_y = center_y - (cube_size // 2)
        for i, color_name in enumerate(parts):
            bgr = COLOR_DB.get(color_name, (128, 128, 128))
            x1 = start_x + i * (cube_size + gap); y1 = top_y; x2 = x1 + cube_size; y2 = y1 + cube_size
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            if i < num_cubes - 1:
                cv2.arrowedLine(img, (x2 + 10, center_y), (x2 + gap - 10, center_y), (200, 200, 200), 4, tipLength=0.3)
    with image_lock: latest_image = img

def update_image_simple_message(main_text, sub_text="", bg_color=(0, 100, 0)):
    global latest_image
    img, w, h = create_image_canvas(bg_color)
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_centered_text(img, main_text, (h//2) - 40, font, 3.0, (255,255,255), 5)
    if sub_text: draw_centered_text(img, sub_text, (h//2) + 60, font, 1.2, (200,200,200), 2)
    with image_lock: latest_image = img

# ==========================================
# FLASK SERVER
# ==========================================
def generate_mjpeg_stream():
    global latest_image
    if latest_image is None: show_instructions()
    while True:
        with image_lock:
            frame = latest_image.copy() if latest_image is not None else np.zeros((600,1024,3), np.uint8)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>TIAGo Controller</title>
        <style>
            body { 
                margin: 0; background: #000; height: 100vh; 
                display: flex; flex-direction: column;
                align-items: center; justify-content: center; 
                font-family: sans-serif;
            }
            .image-container {
                flex: 1; display: flex; align-items: center; justify-content: center;
                width: 100%; overflow: hidden;
            }
            img { max-width: 100%; max-height: 95%; object-fit: contain; }
            .controls { 
                display: flex; flex-direction: column; gap: 20px; 
                align-items: center; justify-content: center; 
                padding-bottom: 40px; width: 100%; min-height: 150px; 
            }
            .btn-base {
                padding: 20px 50px; font-size: 26px; font-weight: bold; color: white;
                border: none; border-radius: 50px; cursor: pointer;
                box-shadow: 0 4px 15px rgba(0,0,0,0.6);
                text-transform: uppercase; min-width: 300px; text-align: center;
                transition: transform 0.1s;
            }
            .btn-base:active { transform: scale(0.98); }
            .btn-start { background-color: #4CAF50; display: block; } 
            .btn-stop { background-color: #f44336; display: none; }   
            .btn-reset { background-color: #2196F3; display: none; }  

            .switch-container {
                display: flex; align-items: center; color: white; 
                font-size: 20px; font-weight: bold; background: #333; 
                padding: 10px 25px; border-radius: 40px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.5);
                margin-bottom: 10px;
            }
            .switch { position: relative; display: inline-block; width: 60px; height: 34px; margin: 0 15px; }
            .switch input { opacity: 0; width: 0; height: 0; }
            .slider {
                position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
                background-color: #ccc; transition: .4s; border-radius: 34px;
            }
            .slider:before {
                position: absolute; content: ""; height: 26px; width: 26px; left: 4px; bottom: 4px;
                background-color: white; transition: .4s; border-radius: 50%;
            }
            input:checked + .slider { background-color: #9C27B0; }
            input:checked + .slider:before { transform: translateX(26px); }
        </style>
        <script>
            let isBusy = false; 
            let checkInterval = null;

            function toggleDifficulty(checkbox) {
                const mode = checkbox.checked ? 'VISUAL' : 'TEXT';
                fetch('/set_difficulty/' + mode);
            }

            function clickStart() {
                if (isBusy) return; isBusy = true;
                document.getElementById("btnStart").style.display = "none";
                document.getElementById("diffPanel").style.display = "none";
                fetch('/trigger_start');
                setTimeout(() => {
                    document.getElementById("btnStop").style.display = "block";
                    isBusy = false;
                }, 11000);
            }

            function clickStop() {
                if (isBusy) return; isBusy = true;
                document.getElementById("btnStop").style.display = "none";
                fetch('/trigger_stop');
                checkInterval = setInterval(() => {
                    fetch('/check_status')
                        .then(response => response.json())
                        .then(data => {
                            if (data.feedback_ready === true) {
                                clearInterval(checkInterval); 
                                document.getElementById("btnReset").style.display = "block";
                                isBusy = false;
                            }
                        })
                        .catch(err => console.error(err));
                }, 1000);
            }

            function clickReset() {
                if (isBusy) return;
                
                // Ocultar botón inmediatamente
                document.getElementById("btnReset").style.display = "none";
                
                // Iniciar proceso
                fetch('/trigger_reset');
                
                // Chequear cuando el robot termina el movimiento
                let resetCheckInterval = setInterval(() => {
                    fetch('/check_status_reset') 
                        .then(response => response.json())
                        .then(data => {
                            if (data.reset_done === true) {
                                clearInterval(resetCheckInterval);
                                // AHORA SÍ: Mostramos los controles porque el robot avisó que acabó
                                document.getElementById("btnStart").style.display = "block";
                                document.getElementById("diffPanel").style.display = "flex";
                            }
                        });
                }, 1000);
            }
        </script>
    </head>
    <body>
        <div class="image-container"><img src="/video_feed"></div>
        <div class="controls">
            <div id="diffPanel" class="switch-container">
                <span>TEXT</span>
                <label class="switch">
                    <input type="checkbox" onchange="toggleDifficulty(this)">
                    <span class="slider"></span>
                </label>
                <span>VISUAL</span>
            </div>
            <button id="btnStart" class="btn-base btn-start" onclick="clickStart()">▶ START EXERCISE</button>
            <button id="btnStop"  class="btn-base btn-stop"  onclick="clickStop()">⏹ STOP EXERCISE</button>
            <button id="btnReset" class="btn-base btn-reset" onclick="clickReset()">↻ RESET EXERCISE</button>
        </div>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed(): return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_difficulty/<mode>')
def set_difficulty(mode):
    global current_difficulty
    current_difficulty = mode
    rospy.loginfo(f"Dificultad cambiada a: {mode}")
    return "OK"

@app.route('/check_status')
def check_status():
    global feedback_ready
    return jsonify({"feedback_ready": feedback_ready})

@app.route('/check_status_reset')
def check_status_reset():
    # Comprobamos si la bandera de movimiento ya se apagó
    global doing_reset_move
    return jsonify({"reset_done": not doing_reset_move})

@app.route('/trigger_start')
def trigger_start():
    global doing_countdown, pending_text, feedback_ready
    doing_countdown = True; pending_text = None
    feedback_ready = False 
    if pub_command: pub_command.publish("start")
    def visual_countdown():
        global doing_countdown, pending_text
        for i in range(10, 0, -1):
            update_image_simple_message(f"STARTING IN {i}...", "Get ready!", bg_color=(200, 100, 0))
            time.sleep(1.0)
        doing_countdown = False
        if pending_text: update_image_with_text(pending_text)
        else: update_image_simple_message("GO!", "Exercise running...", bg_color=(0, 0, 100))
    threading.Thread(target=visual_countdown, daemon=True).start()
    return "OK"

@app.route('/trigger_stop')
def trigger_stop():
    global doing_countdown, feedback_ready
    doing_countdown = False
    feedback_ready = False 
    def delayed_stop_sequence():
        for i in range(5, 0, -1):
            update_image_simple_message(f"FINISHING IN {i}...", "REMOVE HANDS", bg_color=(200, 50, 0))
            time.sleep(1.0)
        if pub_command: pub_command.publish("stop")
        update_image_simple_message("WELL DONE!", "Analyzing...", bg_color=(0, 100, 50))
    threading.Thread(target=delayed_stop_sequence, daemon=True).start()
    return "OK"

@app.route('/trigger_reset')
def trigger_reset():
    global feedback_ready, doing_reset_move
    feedback_ready = False 
    doing_reset_move = True # Activamos el modo "movimiento"
    
    def safe_reset_sequence():
        # 1. Advertencia
        for i in range(5, 0, -1):
            update_image_simple_message(f"RESETTING IN {i}...", "PLEASE REMOVE HANDS", bg_color=(200, 0, 0))
            time.sleep(1.0)
            
        # 2. Ordenar movimiento
        if pub_command: pub_command.publish("reset")
        update_image_simple_message("KEEP HANDS AWAY", "ROBOT MOVING", bg_color=(255, 0, 0))
        
        # NOTA: NO cambiamos doing_reset_move aquí.
        # Esperamos al callback de ROS.

    threading.Thread(target=safe_reset_sequence, daemon=True).start()
    return "OK"

def start_flask(): app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    rospy.init_node('tiago_web_display', anonymous=False)
    web_node = WebDisplayNode()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
    except: ip = "127.0.0.1"
    rospy.loginfo(f"📺 WEB: http://{ip}:{WEB_PORT}")
    flask_thread = threading.Thread(target=start_flask); flask_thread.daemon = True; flask_thread.start()
    try: rospy.spin()
    except KeyboardInterrupt: pass