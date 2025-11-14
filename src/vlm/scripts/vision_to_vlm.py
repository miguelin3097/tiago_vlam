#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
from datetime import datetime

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5 import QtCore, QtGui, QtWidgets

# Servicio (para llamar al nodo de Gemini)
from vlm.srv import GeminiAnalyze, GeminiAnalyzeRequest

DEFAULT_TOPIC = "/xtion/rgb/image_raw"  # ajusta para tu TIAGo (sim/real)

class ImageBuffer(object):
    """Buffer thread-safe del último frame ROS->OpenCV."""
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


class CaptureWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIAGo Photo GUI")
        self.resize(900, 600)

        # Widgets
        self.topicEdit = QtWidgets.QLineEdit(DEFAULT_TOPIC)
        self.btnStart  = QtWidgets.QPushButton("Start")
        self.countdownLbl = QtWidgets.QLabel("Ready")
        self.countdownLbl.setAlignment(QtCore.Qt.AlignCenter)
        font = self.countdownLbl.font()
        font.setPointSize(20)
        self.countdownLbl.setFont(font)

        self.firstImgLbl  = QtWidgets.QLabel("First photo")
        self.secondImgLbl = QtWidgets.QLabel("Second photo")
        for l in (self.firstImgLbl, self.secondImgLbl):
            l.setFixedSize(420, 315)
            l.setFrameShape(QtWidgets.QFrame.Box)
            l.setAlignment(QtCore.Qt.AlignCenter)

        # Layout
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Image topic:"))
        top.addWidget(self.topicEdit, 1)
        top.addWidget(self.btnStart)

        imgs = QtWidgets.QHBoxLayout()
        imgs.addWidget(self.firstImgLbl, 1)
        imgs.addWidget(self.secondImgLbl, 1)

        main = QtWidgets.QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(self.countdownLbl)
        main.addLayout(imgs)

        # Estado
        self.img_buffer = ImageBuffer()
        self.sub = None
        self.timer_ui = QtCore.QTimer(self)
        self.timer_ui.setInterval(100)  # 10 Hz
        self.timer_ui.timeout.connect(self._on_ui_tick)

        self.run_start_time = None
        self.first_done  = False
        self.second_done = False
        self.countdown_secs = 5
        self.total_secs = 60

        self.save_dir = os.path.expanduser("~/.ros/tiago_photos")
        os.makedirs(self.save_dir, exist_ok=True)

        # Para llamar al servicio de Gemini
        self._srv_client = None
        self.gemini_prompt = "¿Qué detectas en la imagen? Describe los objetos y si están ordenados sobre la mesa."
        self.last_first_path = None
        self.last_second_path = None

        # Señales
        self.btnStart.clicked.connect(self.on_start_clicked)

    def start_sub(self):
        topic = self.topicEdit.text().strip()
        if self.sub:
            self.sub.unregister()
            self.sub = None
        self.sub = rospy.Subscriber(topic, Image, self.img_buffer.update, queue_size=1)
        rospy.loginfo(f"Subscribed to {topic}")

    def on_start_clicked(self):
        """Inicia: 5s cuenta atrás -> foto1 ; a los 60s -> foto2 -> llamar servicio Gemini."""
        self.start_sub()
        self.run_start_time = time.time()
        self.first_done  = False
        self.second_done = False
        self.last_first_path = None
        self.last_second_path = None
        self.timer_ui.start()
        self.btnStart.setEnabled(False)
        self.countdownLbl.setText("Countdown: 5")

    def _on_ui_tick(self):
        if self.run_start_time is None:
            return

        elapsed = time.time() - self.run_start_time

        # Cuenta atrás inicial (5 s)
        if not self.first_done:
            left = int(max(0, self.countdown_secs - elapsed))
            self.countdownLbl.setText(f"Countdown: {left}")
            if elapsed >= self.countdown_secs:
                self._capture_first()
                self.first_done = True
                self.countdownLbl.setText("First photo taken. Waiting to 60s…")
                return

        # Segunda foto a los 60 s desde el inicio
        if self.first_done and not self.second_done:
            if elapsed >= self.total_secs:
                self._capture_second()
                self.second_done = True
                self.countdownLbl.setText("Second photo taken. Calling Gemini…")
                self.timer_ui.stop()
                self.btnStart.setEnabled(True)
                self.run_start_time = None
                # Llamar al servicio de Gemini
                self._call_gemini_service()

    def _save_and_show(self, img, which="first"):
        """Guarda imagen y la muestra. Devuelve la ruta guardada."""
        import cv2
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(self.save_dir, f"{which}_{ts}.png")
        cv2.imwrite(fn, img)
        rospy.loginfo(f"Saved {which} photo -> {fn}")
        # Mostrar en QLabel
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qimg)
        target = self.firstImgLbl if which == "first" else self.secondImgLbl
        target.setPixmap(pix.scaled(target.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        return fn

    def _capture_first(self):
        frame = self.img_buffer.get()
        if frame is None:
            rospy.logwarn("No frame available for FIRST photo.")
            self.countdownLbl.setText("No image available! (FIRST)")
            return
        self.last_first_path = self._save_and_show(frame, "first")

    def _capture_second(self):
        frame = self.img_buffer.get()
        if frame is None:
            rospy.logwarn("No frame available for SECOND photo.")
            self.countdownLbl.setText("No image available! (SECOND)")
            return
        self.last_second_path = self._save_and_show(frame, "second")

    def _ensure_service(self):
        if self._srv_client is None:
            rospy.wait_for_service("/gemini_vision/analyze_images", timeout=10.0)
            self._srv_client = rospy.ServiceProxy("/gemini_vision/analyze_images", GeminiAnalyze)

    def _call_gemini_service(self):
        try:
            if not (self.last_first_path and self.last_second_path):
                self.countdownLbl.setText("Faltan rutas para Gemini")
                return
            self._ensure_service()
            req = GeminiAnalyzeRequest(
                image1_path=self.last_first_path,
                image2_path=self.last_second_path,
                prompt=self.gemini_prompt
            )
            self.countdownLbl.setText("Consultando Gemini…")
            resp = self._srv_client(req)
            if resp.success:
                shown = (resp.answer[:250] + "…") if len(resp.answer) > 250 else resp.answer
                self.countdownLbl.setText("Gemini: " + shown)
                rospy.loginfo("Gemini answer:\n" + resp.answer)
            else:
                msg = "Gemini ERROR: " + resp.error
                self.countdownLbl.setText(msg)
                rospy.logerr(msg)
        except Exception as e:
            rospy.logerr("Excepción al llamar servicio Gemini: %s", e)
            self.countdownLbl.setText("Excepción llamando a Gemini")


def main():
    import sys
    rospy.init_node("tiago_photo_gui", anonymous=False, disable_signals=True)
    app = QtWidgets.QApplication(sys.argv)
    w = CaptureWindow()
    w.show()
    # Timer “no-op” para que Ctrl+C funcione con Qt
    ros_timer = QtCore.QTimer()
    ros_timer.timeout.connect(lambda: None)
    ros_timer.start(50)
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

