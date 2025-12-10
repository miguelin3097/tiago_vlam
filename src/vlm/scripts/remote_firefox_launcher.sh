#!/bin/bash

# CONFIGURATION
ROBOT_USER="pal"
ROBOT_IP="192.168.50.178" # <--- SET TIAGO'S REAL IP HERE OR USE ITS HOSTNAME
URL="http://192.168.50.234:5000"

# --- CLEANUP FUNCTION (Executed on Ctrl+C) ---
cleanup() {
    echo ""
    echo "Termination signal received."
    echo "Closing Firefox on the robot..."
    
    # Connect via standard SSH (passwordless) and kill the process
    ssh -o StrictHostKeyChecking=no $ROBOT_USER@$ROBOT_IP "pkill -f firefox"
    
    echo "Firefox closed."
    exit 0
}

# --- TRAP SIGNALS ---
# If ROS or the user sends Ctrl+C (SIGINT), execute 'cleanup'
trap cleanup SIGINT SIGTERM

echo "Launching Firefox on the robot ($ROBOT_IP)..."

# 1. Open Firefox (Standard SSH with keys)
# -f: Launches in background within SSH to prevent hanging/waiting
ssh -o StrictHostKeyChecking=no -f $ROBOT_USER@$ROBOT_IP "export DISPLAY=:0 && firefox -kiosk $URL > /dev/null 2>&1 &"

echo "Browser open. Keeping node active..."
echo "Press Ctrl+C to close everything."

# 2. WAIT LOOP
# We need this infinite loop so the script doesn't terminate immediately.
# If the script terminates, ROS assumes the node has died.
# By keeping it alive, ROS can send the termination signal when needed.
while true; do
    sleep 1
done