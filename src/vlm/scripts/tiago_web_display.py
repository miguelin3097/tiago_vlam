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
from cv_bridge import CvBridge, CvBridgeError

# ==========================================
# CONFIGURACIÓN
# ==========================================
TOPIC_NAME = '/robot/xdisplay' 
WEB_PORT = 5000

# ==========================================
# ESTADO GLOBAL
# ==========================================
app = Flask(__name__)
latest_image = None
image_lock = threading.Lock()

class WebDisplayNode:
    def __init__(self):
        # Nota: No iniciamos el nodo aquí, lo hacemos en el main para controlar las señales
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(TOPIC_NAME, Image, self.image_callback)
        
    def image_callback(self, msg):
        global latest_image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with image_lock:
                latest_image = cv_img     
        except CvBridgeError as e:
            rospy.logerr(f"Error CvBridge: {e}")

# ==========================================
# FLASK (Servidor Web)
# ==========================================
def generate_mjpeg_stream():
    global latest_image
    while True:
        with image_lock:
            if latest_image is None:
                frame = np.zeros((600, 1024, 3), dtype=np.uint8)
                cv2.putText(frame, "WAITING SEQUENCE...", (300, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.putText(frame, f"Topic: {TOPIC_NAME}", (350, 400), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
            else:
                frame = latest_image.copy()

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>TIAGo Display</title>
        <style>
            body { margin: 0; background: #000; height: 100vh; display: flex; 
                   justify-content: center; align-items: center; overflow: hidden; }
            img { max-width: 100%; max-height: 100%; object-fit: contain; }
        </style>
    </head>
    <body>
        <img src="/video_feed">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_server():
    """Función para arrancar Flask en un hilo aparte."""
    # use_reloader=False es CRÍTICO para que no cree subprocesos que bloqueen el cierre
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)

# ==========================================
# MAIN (Punto de entrada)
# ==========================================
if __name__ == '__main__':
    # 1. Iniciar el Nodo ROS
    rospy.init_node('tiago_web_display', anonymous=False)

    # Obtener IP para mostrarla
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "127.0.0.1"

    rospy.loginfo(f"📺 SERVIDOR WEB INICIADO")
    rospy.loginfo(f"🌍 URL: http://{ip}:{WEB_PORT}")

    # 2. Instanciar la clase lógica
    web_node = WebDisplayNode()

    # 3. Arrancar Flask en un HILO DAEMON
    # daemon=True significa: "Si el programa principal se cierra, mata este hilo inmediatamente"
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True 
    flask_thread.start()

    # 4. Bucle principal de ROS
    # rospy.spin() bloqueará aquí hasta que pulses Ctrl+C
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Cerrando...")
    
    # Al salir de rospy.spin(), el programa termina y el hilo Flask muere solo.