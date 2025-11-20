#!/usr/bin/env python3
import sys
import rospy
import math
import moveit_commander
import time
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QGroupBox, QMessageBox, QTextEdit, QFrame, QSlider)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QDoubleValidator, QFont

# --- CONFIGURATION ---
READY_JOINTS = [0.629, 0.009, 1.637, 1.168, 1.093, -1.339, -1.506]
DEFAULT_ROT = {'r': 90.0, 'p': 30.0, 'y': 90.0}

# --- WORKER THREAD ---
class WorkerThread(QThread):
    finished_signal = pyqtSignal(bool, str)
    log_signal = pyqtSignal(str)

    def __init__(self, task_function, *args):
        super().__init__()
        self.task_function = task_function
        self.args = args

    def run(self):
        try:
            success, msg = self.task_function(self.log_signal, *self.args)
            self.finished_signal.emit(success, msg)
        except Exception as e:
            self.finished_signal.emit(False, f"Critical Error: {str(e)}")

# --- ROBOT CONTROLLER ---
class RobotController:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        try:
            rospy.init_node('tiago_qt_gui', anonymous=True)
        except rospy.exceptions.ROSException:
            pass
        
        self.robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("arm_right")
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper_right")
        self.torso_group = moveit_commander.MoveGroupCommander("torso")
        print(f"\n\n[INFO] REFERENCE FRAME {self.arm_group.get_planning_frame()}\n\n")
        
        self.arm_group.set_planning_time(5.0)
        self.arm_group.set_num_planning_attempts(10)
        self.arm_group.set_goal_position_tolerance(0.005)
        self.arm_group.set_goal_orientation_tolerance(0.05)
        
        self.stop_flag = False

    def check_stop(self):
        return self.stop_flag

    def smart_sleep(self, duration):
        steps = int(duration / 0.05)
        for _ in range(steps):
            if self.stop_flag: return False
            time.sleep(0.05)
        return True

    def _create_pose(self, x, y, z, r_deg, p_deg, y_deg):
        planning_frame = self.arm_group.get_planning_frame()
        
        pose = PoseStamped()
        pose.header.frame_id = planning_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        q = quaternion_from_euler(math.radians(r_deg), math.radians(p_deg), math.radians(y_deg))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def _move_to_pose(self, pose):
        if self.stop_flag: return False
        self.arm_group.set_pose_target(pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        if self.stop_flag: return False
        return success

    def _move_to_joints(self, joint_values):
        if self.stop_flag: return False
        joint_names = self.arm_group.get_active_joints()
        if len(joint_values) != len(joint_names): return False
        target = dict(zip(joint_names, joint_values))
        self.arm_group.set_joint_value_target(target)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        if self.stop_flag: return False
        return success

    def _move_gripper(self, state):
        if self.stop_flag: return False
        vals = [0.044, 0.044] if state == "open" else [0.005, 0.005]
        self.gripper_group.set_joint_value_target(vals)
        return self.gripper_group.go(wait=True)

    def get_robot_status(self, logger):
        joints = self.arm_group.get_current_joint_values()
        joints_rounded = [round(j, 3) for j in joints]
        pose = self.arm_group.get_current_pose().pose
        p = pose.position
        o = pose.orientation
        torso_val = self.torso_group.get_current_joint_values()[0]
        q_list = [o.x, o.y, o.z, o.w]
        (roll_rad, pitch_rad, yaw_rad) = euler_from_quaternion(q_list)
        roll_deg = round(math.degrees(roll_rad), 1)
        pitch_deg = round(math.degrees(pitch_rad), 1)
        yaw_deg = round(math.degrees(yaw_rad), 1)

        msg = (f"\n--- ROBOT STATUS ---\n"
               f"TORSO HEIGHT: {torso_val:.3f} m\n"
               f"JOINTS: {joints_rounded}\n"
               f"POS: X={p.x:.3f}, Y={p.y:.3f}, Z={p.z:.3f}\n"
               f"ROT: R={roll_deg}, P={pitch_deg}, Y={yaw_deg}\n"
               f"--------------------")
        logger.emit(msg)
        return True, "Status retrieved."

    def stop_robot(self):
        self.stop_flag = True
        self.arm_group.stop()
        self.gripper_group.stop()
        self.torso_group.stop()
        return True, "!!! STOP EXECUTED !!!"

    def execute_torso_move(self, logger, height):
        self.stop_flag = False
        logger.emit(f"Moving Torso to {height:.2f}m...")
        self.torso_group.set_joint_value_target([height])
        success = self.torso_group.go(wait=True)
        self.torso_group.stop()
        return success, f"Torso at {height:.2f}m"

    def go_home_safe(self, logger):
        self.stop_flag = False
        logger.emit("Moving to Safe Home...")
        safe_values = [0.20, -0.50, -0.20, 1.50, -1.00, 1.00, 0.00]
        if not self._move_to_joints(safe_values): return False, "Stopped."
        return True, "Robot at Safe HOME."

    def go_to_ready_pose(self, logger):
        self.stop_flag = False
        logger.emit("Moving to READY JOINTS...")
        if not self._move_to_joints(READY_JOINTS): return False, "Interrupted."
        return True, "Robot at READY."

    def execute_manual_move(self, logger, x, y, z, r, p, yw):
        self.stop_flag = False
        logger.emit(f"Manual Move: ({x}, {y}, {z})")
        pose = self._create_pose(x, y, z, r, p, yw)
        if not self._move_to_pose(pose): return False, "Interrupted."
        return True, "Movement finished."

    def execute_gripper(self, logger, state):
        self.stop_flag = False
        logger.emit(f"Gripper: {state}")
        if not self._move_gripper(state): return False, "Interrupted."
        return True, f"Gripper {state}"

    def execute_pick_place(self, logger, px, py, pz, lx, ly, lz, roll, pitch, yaw):
        self.stop_flag = False
        OFFSET_Z = 0.15 
        logger.emit(f"1. Starting Sequence (Rot: {roll}, {pitch}, {yaw})...")

        if self.check_stop(): return False, "Aborted."
        if not self._move_to_joints(READY_JOINTS): return False, "Stop at Start."
        self._move_gripper("open")

        # PICK
        logger.emit(f"2. Pick ({px}, {py})...")
        if self.check_stop(): return False, "Aborted."
        pose_pick_app = self._create_pose(px, py, pz + OFFSET_Z, roll, pitch, yaw)
        if not self._move_to_pose(pose_pick_app): return False, "Stop Pick App."

        if self.check_stop(): return False, "Aborted."
        pose_pick = self._create_pose(px, py, pz, roll, pitch, yaw)
        if not self._move_to_pose(pose_pick): return False, "Stop Pick Down."

        logger.emit("   > Closing gripper...")
        if self.check_stop(): return False, "Aborted."
        self._move_gripper("close")
        if not self.smart_sleep(0.5): return False, "Stop."

        logger.emit("   > Moving Up...")
        if self.check_stop(): return False, "Aborted."
        if not self._move_to_pose(pose_pick_app): return False, "Stop Pick Up."

        # PLACE
        logger.emit(f"3. Place ({lx}, {ly})...")
        if self.check_stop(): return False, "Aborted."
        pose_place_app = self._create_pose(lx, ly, lz + OFFSET_Z, roll, pitch, yaw)
        if not self._move_to_pose(pose_place_app): return False, "Stop Place App."

        if self.check_stop(): return False, "Aborted."
        pose_place = self._create_pose(lx, ly, lz, roll, pitch, yaw)
        if not self._move_to_pose(pose_place): return False, "Stop Place Down."

        logger.emit("   > Opening gripper...")
        if self.check_stop(): return False, "Aborted."
        self._move_gripper("open")
        if not self.smart_sleep(0.5): return False, "Stop."

        logger.emit("   > Moving Up...")
        if self.check_stop(): return False, "Aborted."
        if not self._move_to_pose(pose_place_app): return False, "Stop Place Up."

        logger.emit("4. Returning to Ready...")
        if not self._move_to_joints(READY_JOINTS): return False, "Stop at End."

        return True, "Cycle OK."

# --- GUI ---
class TiagoControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = RobotController()
        self.lockable_widgets = [] 
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TIAGo Master Control v14 - FINAL")
        self.setGeometry(100, 100, 550, 950)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-family: Arial;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        title = QLabel("TIAGo Command Center")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #00d0ff; margin: 5px;")
        main_layout.addWidget(title)

        # TOP BUTTONS
        top_frame = QFrame()
        top_layout = QHBoxLayout(top_frame)
        btn_home = QPushButton("HOME")
        btn_home.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        btn_home.clicked.connect(self.action_go_home)
        btn_ready = QPushButton("READY (Joints)")
        btn_ready.setStyleSheet("background-color: #2196F3; font-weight: bold;")
        btn_ready.clicked.connect(self.action_go_ready)
        btn_open = QPushButton("OPEN")
        btn_open.setStyleSheet("background-color: #555; border: 1px solid #777;")
        btn_open.clicked.connect(lambda: self.action_gripper("open"))
        btn_close = QPushButton("CLOSE")
        btn_close.setStyleSheet("background-color: #555; border: 1px solid #777;")
        btn_close.clicked.connect(lambda: self.action_gripper("close"))
        top_layout.addWidget(btn_home)
        top_layout.addWidget(btn_ready)
        top_layout.addWidget(btn_open)
        top_layout.addWidget(btn_close)
        main_layout.addWidget(top_frame)
        self.lockable_widgets.append(top_frame)

        # TORSO SECTION
        self.create_torso_section(main_layout)

        # MANUAL SECTION
        self.create_manual_section(main_layout)

        # AUTO SECTION
        self.create_pick_place_section(main_layout)

        # STOP BUTTON
        self.btn_stop = QPushButton("EMERGENCY STOP")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 20px; font-size: 14pt; margin-top: 15px; border-radius: 10px;")
        self.btn_stop.clicked.connect(self.action_stop)
        main_layout.addWidget(self.btn_stop)

        # LOG
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #101010; color: #00ff00; font-family: Monospace;")
        self.log_box.setFixedHeight(200)
        main_layout.addWidget(self.log_box)

        self.status_label = QLabel("Ready.")
        main_layout.addWidget(self.status_label)

    def create_torso_section(self, parent):
        group = QGroupBox("Torso Control")
        group.setStyleSheet("QGroupBox { border: 1px solid #FFC107; margin-top: 5px; font-weight: bold; }")
        layout = QVBoxLayout()
        
        self.lbl_torso = QLabel("Height: 0.00 m")
        layout.addWidget(self.lbl_torso)
        
        self.slider_torso = QSlider(Qt.Horizontal)
        self.slider_torso.setMinimum(0)
        self.slider_torso.setMaximum(35)
        self.slider_torso.setValue(0)
        self.slider_torso.setTickInterval(5)
        self.slider_torso.setTickPosition(QSlider.TicksBelow)
        self.slider_torso.valueChanged.connect(self.update_torso_label)
        layout.addWidget(self.slider_torso)
        
        btn = QPushButton("MOVE TORSO")
        btn.setStyleSheet("background-color: #FFC107; color: black; font-weight: bold;")
        btn.clicked.connect(self.action_move_torso)
        layout.addWidget(btn)
        
        group.setLayout(layout)
        parent.addWidget(group)
        self.lockable_widgets.append(group)

    def update_torso_label(self, value):
        self.lbl_torso.setText(f"Height: {value/100.0:.2f} m")

    def create_manual_section(self, parent):
        group = QGroupBox("Manual Control & Info")
        group.setStyleSheet("QGroupBox { border: 1px solid #777; margin-top: 10px; font-weight: bold; }")
        layout = QVBoxLayout()
        
        # Inputs X, Y, Z
        r1 = QHBoxLayout()
        self.man_x = self.create_input("X:", "0.55", r1)
        self.man_y = self.create_input("Y:", "-0.10", r1)
        self.man_z = self.create_input("Z:", "1.00", r1)
        layout.addLayout(r1)

        # Inputs R, P, Y
        r2 = QHBoxLayout()
        self.man_r = self.create_input("Roll:", "90.0", r2)
        self.man_p = self.create_input("Pitch:", "90.0", r2)
        self.man_yw = self.create_input("Yaw:", "0.0", r2)
        layout.addLayout(r2)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_move = QPushButton("MOVE TO POSE")
        btn_move.setStyleSheet("background-color: #FF9800; color: black; font-weight: bold;")
        btn_move.clicked.connect(self.action_manual_move)
        
        btn_status = QPushButton("GET ROBOT STATUS")
        btn_status.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        btn_status.clicked.connect(self.action_get_status)

        btn_layout.addWidget(btn_move)
        btn_layout.addWidget(btn_status)
        layout.addLayout(btn_layout)

        group.setLayout(layout)
        parent.addWidget(group)
        self.lockable_widgets.append(group)

    def create_pick_place_section(self, parent):
        group = QGroupBox("Automatic Sequence")
        group.setStyleSheet("QGroupBox { border: 1px solid #00d0ff; margin-top: 10px; font-weight: bold; }")
        layout = QVBoxLayout()
        
        # PICK Row
        f_pick = QFrame()
        f_pick.setStyleSheet("background-color: #333;")
        lp = QHBoxLayout(f_pick)
        lp.addWidget(QLabel("PICK:"))
        self.pp_px = self.create_input("X", "0.60", lp)
        self.pp_py = self.create_input("Y", "-0.20", lp)
        self.pp_pz = self.create_input("Z", "0.75", lp)
        layout.addWidget(f_pick)

        # PLACE Row
        f_place = QFrame()
        f_place.setStyleSheet("background-color: #333;")
        ll = QHBoxLayout(f_place)
        ll.addWidget(QLabel("PLACE:"))
        self.pp_lx = self.create_input("X", "0.60", ll)
        self.pp_ly = self.create_input("Y", "0.20", ll)
        self.pp_lz = self.create_input("Z", "0.75", ll)
        layout.addWidget(f_place)

        # ROTATION Row
        f_rot = QFrame()
        f_rot.setStyleSheet("background-color: #444; border-radius: 5px;")
        lr = QHBoxLayout(f_rot)
        lr.addWidget(QLabel("GRASP ROTATION:"))
        self.pp_r = self.create_input("R", str(DEFAULT_ROT['r']), lr)
        self.pp_p = self.create_input("P", str(DEFAULT_ROT['p']), lr)
        self.pp_y = self.create_input("Y", str(DEFAULT_ROT['y']), lr)
        layout.addWidget(f_rot)

        btn = QPushButton("START SEQUENCE")
        btn.setStyleSheet("background-color: #00BCD4; color: black; font-weight: bold;")
        btn.clicked.connect(self.action_start_sequence)
        layout.addWidget(btn)
        group.setLayout(layout)
        parent.addWidget(group)
        self.lockable_widgets.append(group)

    def create_input(self, label, default, layout):
        l = QLabel(label)
        l.setAlignment(Qt.AlignRight | Qt.AlignVCenter) # Alineado a la derecha y verticalmente
        i = QLineEdit(default)
        i.setFixedWidth(60)
        i.setStyleSheet("background-color: #555; color: white; border: none; padding: 2px;")
        layout.addWidget(l)
        layout.addWidget(i)
        layout.addSpacing(10) # Espacio entre grupos
        return i

    def log(self, msg):
        self.log_box.append(f"> {msg}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_busy(self, busy):
        for widget in self.lockable_widgets:
            widget.setEnabled(not busy)

    def on_finished(self, success, msg):
        self.set_busy(False)
        self.status_label.setText(msg)
        if "Status" not in msg:
            self.log(f"END: {msg}")

    def action_stop(self):
        self.log("!!! STOP PRESSED !!!")
        self.controller.stop_robot()

    def action_move_torso(self):
        height = self.slider_torso.value() / 100.0
        self.set_busy(True)
        self.worker = WorkerThread(self.controller.execute_torso_move, height)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def action_get_status(self):
        self.set_busy(True)
        self.worker = WorkerThread(self.controller.get_robot_status)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def action_go_home(self):
        self.set_busy(True)
        self.worker = WorkerThread(self.controller.go_home_safe)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def action_go_ready(self):
        self.set_busy(True)
        self.worker = WorkerThread(self.controller.go_to_ready_pose)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def action_gripper(self, state):
        self.set_busy(True)
        self.worker = WorkerThread(self.controller.execute_gripper, state)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def action_manual_move(self):
        try:
            x = float(self.man_x.text())
            y = float(self.man_y.text())
            z = float(self.man_z.text())
            r = float(self.man_r.text())
            p = float(self.man_p.text())
            yw = float(self.man_yw.text())
            self.set_busy(True)
            self.worker = WorkerThread(self.controller.execute_manual_move, x, y, z, r, p, yw)
            self.worker.log_signal.connect(self.log)
            self.worker.finished_signal.connect(self.on_finished)
            self.worker.start()
        except: pass

    def action_start_sequence(self):
        try:
            px = float(self.pp_px.text())
            py = float(self.pp_py.text())
            pz = float(self.pp_pz.text())
            lx = float(self.pp_lx.text())
            ly = float(self.pp_ly.text())
            lz = float(self.pp_lz.text())
            roll = float(self.pp_r.text())
            pitch = float(self.pp_p.text())
            yaw = float(self.pp_y.text())

            self.set_busy(True)
            self.worker = WorkerThread(self.controller.execute_pick_place, px, py, pz, lx, ly, lz, roll, pitch, yaw)
            self.worker.log_signal.connect(self.log)
            self.worker.finished_signal.connect(self.on_finished)
            self.worker.start()
        except Exception as e:
            print(f"Input error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TiagoControlWindow()
    win.show()
    sys.exit(app.exec_())