#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import subprocess
from std_msgs.msg import String

# --- CONFIGURACIÓN ---
ROBOT_IP = "192.168.50.178"  # Pon la IP de tu TIAGo
ROBOT_USER = "pal"          # Usuario
# ---------------------

class SSHOneLinerTTS:
    def __init__(self):
        rospy.init_node('tiago_ssh_oneliner')
        
        # Suscriptor al topic
        self.sub = rospy.Subscriber('/tiago_habla', String, self.callback)
        
        rospy.loginfo("--- TTS REMOTO ONE-LINER LISTO ---")
        rospy.loginfo(f"Enviando comandos a: {ROBOT_USER}@{ROBOT_IP}")

    def callback(self, msg):
        texto = msg.data
        rospy.loginfo(f"Ordenando decir: '{texto}'")
        
        # 1. Limpieza de texto para evitar romper el comando bash
        # Quitamos comillas simples y dobles para que no corten el string del SSH
        safe_text = texto.replace("'", "").replace('"', "").replace("|", ";")
        
        # 2. Construcción del 'One-Liner'
        # Secuencia: Generar -> Reproducir -> Borrar
        remote_cmd = (
            f"pico2wave -w /tmp/remote_tts.wav '{safe_text}' && "
            f"aplay -q /tmp/remote_tts.wav && "
            f"rm /tmp/remote_tts.wav"
        )
        
        # 3. Comando SSH completo
        # Usamos -o StrictHostKeyChecking=no para evitar preguntas de seguridad
        full_ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            f"{ROBOT_USER}@{ROBOT_IP}",
            remote_cmd
        ]
        
        try:
            # Ejecutamos y esperamos a que termine (call es bloqueante)
            # Si quieres que sea asíncrono (no bloquear el nodo), usa subprocess.Popen
            subprocess.call(full_ssh_cmd)
            
        except Exception as e:
            rospy.logerr(f"Error lanzando SSH: {e}")

if __name__ == '__main__':
    try:
        SSHOneLinerTTS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass