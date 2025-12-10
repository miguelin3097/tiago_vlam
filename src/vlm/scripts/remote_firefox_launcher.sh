#!/bin/bash

# CONFIGURACIÓN
ROBOT_USER="pal"
ROBOT_IP="192.168.50.178" # <--- PON LA IP REAL DEL TIAGO AQUÍ O USA SU HOSTNAME
URL="http://192.168.50.234:5000"

# --- FUNCIÓN DE LIMPIEZA (Se ejecuta al hacer Ctrl+C) ---
cleanup() {
    echo ""
    echo "Recibida señal de cierre."
    echo "Cerrando Firefox en el robot..."
    
    # Nos conectamos por SSH normal (sin contraseña) y matamos el proceso
    ssh -o StrictHostKeyChecking=no $ROBOT_USER@$ROBOT_IP "pkill -f firefox"
    
    echo "Firefox cerrado."
    exit 0
}

# --- ATRAPAR SEÑALES ---
# Si ROS o tú enviáis Ctrl+C (SIGINT), ejecutamos 'cleanup'
trap cleanup SIGINT SIGTERM

echo "🚀 Lanzando Firefox en el robot ($ROBOT_IP)..."

# 1. Abrimos Firefox (SSH normal con claves)
# -f: Lo lanza en background en el SSH para que no se quede colgado esperando
ssh -o StrictHostKeyChecking=no -f $ROBOT_USER@$ROBOT_IP "export DISPLAY=:0 && firefox -kiosk $URL > /dev/null 2>&1 &"

echo "Navegador abierto. Manteniendo nodo activo..."
echo "Pulsa Ctrl+C para cerrar todo."

# 2. BUCLE DE ESPERA
# Necesitamos este bucle infinito para que el script no termine inmediatamente.
# Si el script termina, ROS cree que el nodo ha muerto.
# Al mantenerlo vivo, ROS puede enviarle la señal de cierre cuando tú quieras.
while true; do
    sleep 1
done