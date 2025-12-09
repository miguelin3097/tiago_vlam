#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import threading
import time
import socket
import numpy as np
from flask import Flask, Response
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

# ==========================================
# CONFIGURACIÓN
# ==========================================
TOPIC_IMG = '/robot/xdisplay'
TOPIC_TEXT = '/web_display/text'
TOPIC_CMD = '/web/command'
WEB_PORT = 5000

# ==========================================
# ESTADO GLOBAL
# ==========================================
app = Flask(__name__)
latest_image = None
image_lock = threading.Lock()
pub_command = None

# ==========================================
# CLASE ROS (NODO)
# ==========================================
class WebDisplayNode:
    def __init__(self):
        # Nota: El nodo se inicia en el main, no aquí
        self.bridge = CvBridge()
        
        # Suscriptores
        self.sub_img = rospy.Subscriber(TOPIC_IMG, Image, self.image_callback)
        self.sub_text = rospy.Subscriber(TOPIC_TEXT, String, self.text_callback)
        
        # Publicador (Para enviar START/STOP al otro programa)
        global pub_command
        pub_command = rospy.Publisher(TOPIC_CMD, String, queue_size=1)

    def image_callback(self, msg):
        global latest_image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with image_lock:
                latest_image = cv_img
        except CvBridgeError as e:
            rospy.logerr(f"Error bridge: {e}")

    def text_callback(self, msg):
        """Genera una imagen bonita a partir del texto de la secuencia."""
        raw_text = msg.data
        update_image_with_text(raw_text)

# ==========================================
# UTILIDADES GRÁFICAS
# ==========================================
def create_image_canvas(bg_color=(0, 0, 0)):
    """Crea un lienzo vacío del tamaño de la pantalla."""
    h, w = 600, 1024
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bg_color # Rellenar color fondo
    return img, w, h

def update_image_with_text(text, bg_color=(0, 0, 0), title="SEQUENCE:"):
    """Helper para crear una imagen con texto y actualizar la global."""
    global latest_image
    
    img, w, h = create_image_canvas(bg_color)
    
    # Formatear texto (flechas)
    if "," in text:
        parts = [p.strip().upper() for p in text.split(',')]
        formatted_text = " -> ".join(parts)
    else:
        formatted_text = text.upper()

    # Config Fuente
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 4
    
    # Calcular centro
    (text_w, text_h), _ = cv2.getTextSize(formatted_text, font, font_scale, thickness)
    
    # Ajustar si es muy largo
    if text_w > (w - 60):
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(formatted_text, font, font_scale, thickness)

    x = (w - text_w) // 2
    y = (h + text_h) // 2

    # Dibujar
    if title:
        cv2.putText(img, title, (50, 80), font, 1, (200, 200, 200), 2)
    
    cv2.putText(img, formatted_text, (x, y), font, font_scale, (255, 255, 255), thickness)
    cv2.line(img, (x, y + 25), (x + text_w, y + 25), (0, 255, 255), 2)

    with image_lock:
        latest_image = img

def update_image_simple_message(main_text, sub_text="", bg_color=(0, 100, 0)):
    """Muestra un mensaje simple (Ej: WELL DONE)."""
    global latest_image
    img, w, h = create_image_canvas(bg_color)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Texto Principal
    (tw, th), _ = cv2.getTextSize(main_text, font, 3.0, 5)
    cv2.putText(img, main_text, ((w-tw)//2, (h+th)//2 - 40), font, 3.0, (255,255,255), 5)
    
    # Subtexto
    if sub_text:
        (tw2, th2), _ = cv2.getTextSize(sub_text, font, 1.2, 2)
        cv2.putText(img, sub_text, ((w-tw2)//2, (h+th)//2 + 60), font, 1.2, (200,200,200), 2)

    with image_lock:
        latest_image = img

# ==========================================
# FLASK SERVER
# ==========================================
def generate_mjpeg_stream():
    global latest_image
    while True:
        with image_lock:
            if latest_image is None:
                img, w, h = create_image_canvas()
                cv2.putText(img, "READY TO START...", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
                frame = img
            else:
                frame = latest_image.copy()

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
            body { margin: 0; background: #000; height: 100vh; display: flex; flex-direction: column;
                   justify-content: center; align-items: center; overflow: hidden; font-family: sans-serif;}
            img { max-width: 100%; max-height: 100%; object-fit: contain; }
            
            .btn-toggle {
                position: absolute; bottom: 30px; padding: 25px 50px;
                font-size: 28px; font-weight: bold; color: white;
                background-color: #4CAF50; border: none; border-radius: 60px;
                cursor: pointer; box-shadow: 0 6px 20px rgba(0,0,0,0.6);
                transition: all 0.2s; text-transform: uppercase;
            }
            .btn-toggle:active { transform: scale(0.95); }
            .btn-stop { background-color: #f44336 !important; box-shadow: 0 0 15px #f44336; }
        </style>
        <script>
            let isRunning = false;
            function toggleState() {
                const btn = document.getElementById("mainBtn");
                if (!isRunning) {
                    fetch('/trigger_start').then(() => {
                        isRunning = true;
                        btn.innerText = "⏹ STOP EXERCISE";
                        btn.classList.add("btn-stop");
                    });
                } else {
                    fetch('/trigger_stop').then(() => {
                        isRunning = false;
                        btn.innerText = "▶ START EXERCISE";
                        btn.classList.remove("btn-stop");
                    });
                }
            }
        </script>
    </head>
    <body>
        <img src="/video_feed">
        <button id="mainBtn" class="btn-toggle" onclick="toggleState()">▶ START EXERCISE</button>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/trigger_start')
def trigger_start():
    if pub_command:
        pub_command.publish("start")
        # Mostrar mensaje inicial
        update_image_simple_message("GET READY!", "Exercise starting...", bg_color=(0, 0, 100))
        rospy.loginfo("WEB: START triggered")
    return "OK"

@app.route('/trigger_stop')
def trigger_stop():
    if pub_command:
        pub_command.publish("stop")
        # --- AQUÍ ESTÁ EL MENSAJE DE FELICITACIÓN ---
        update_image_simple_message("WELL DONE!", "Great job!", bg_color=(0, 128, 0))
        rospy.loginfo("WEB: STOP triggered")
    return "OK"

# ==========================================
# MAIN Y CIERRE SEGURO
# ==========================================
def start_flask():
    # use_reloader=False evita subprocesos conflictivos
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    # 1. Iniciar ROS
    rospy.init_node('tiago_web_display', anonymous=False)
    
    # Mostrar IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "127.0.0.1"
    
    rospy.loginfo(f"📺 SERVIDOR WEB ACTIVO")
    rospy.loginfo(f"🌍 URL: http://{ip}:{WEB_PORT}")

    # 2. Iniciar Lógica del Nodo
    web_node = WebDisplayNode()

    # 3. Arrancar Flask en HILO DAEMON
    # Esto es la clave para que Control+C funcione
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True 
    flask_thread.start()

    # 4. Mantener vivo el programa hasta Control+C
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Cerrando servidor web...")
    
    # Al salir de aquí, el hilo daemon muere automáticamente.