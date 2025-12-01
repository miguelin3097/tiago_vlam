#!/usr/bin/env python3
import sys
import rospy
import actionlib
import math
import time
import moveit_commander
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped

# PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui

# Mensajes ROS
from vlm.msg import PickUpAction, PickUpGoal, PickUpResult, PickUpFeedback

# --- CONFIGURACIÓN ---
READY_JOINTS = [0.629, 0.009, 1.637, 1.168, 1.093, -1.339, -1.506]
# (Opcional) Puedes descomentar esto si quieres tu pose de REST
# REST_JOINTS  = [-1.080, 1.480, 2.872, 1.584, -1.599, 1.358, -0.005]

DEFAULT_ROT = {'r': 90.0, 'p': 30.0, 'y': 90.0} # Top-Down para cubos

# --- CLASE WORKER ---
class WorkerThread(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(bool, str)
    
    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        try:
            success, msg = self.func(*self.args)
            self.finished_signal.emit(success, msg)
        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}")

# --- CONTROLADOR DEL ROBOT ---
class RobotController(QtCore.QObject):
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        
        self.robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("arm_right")
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper_right")
        # SIN TORSO
        
        self.arm_group.set_planning_time(10.0)
        self.arm_group.set_num_planning_attempts(5)
        self.arm_group.set_goal_position_tolerance(0.005)
        self.arm_group.set_goal_orientation_tolerance(0.05)

        self.stop_flag = False
        self.is_busy = False 

        # Action Server
        self.server = actionlib.SimpleActionServer('/vlm/pickup', PickUpAction, self.execute_action_cb, False)
        self.server.start()
        self.log_signal.emit("PickUp Server READY (No Torso Control).")

    # --- UTILS ---
    def _feedback(self, msg):
        self.server.publish_feedback(PickUpFeedback(current_state=msg))
        self.log_signal.emit(f"[ACTION] {msg}")

    def check_stop(self):
        if self.server.is_active() and self.server.is_preempt_requested():
            self.log_signal.emit("Action Preempted!")
            self.server.set_preempted()
            self.stop_robot()
            return True
        return self.stop_flag

    def stop_robot(self):
        self.stop_flag = True
        self.arm_group.stop()
        self.gripper_group.stop()
        self.log_signal.emit("!!! STOP EXECUTED !!!")
        return True, "Stopped"

    # --- MOVIMIENTOS ---
    def _move_to_pose(self, x, y, z, r, p, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.arm_group.get_planning_frame()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        self._feedback(f"Moving to {x}, {y}, {z}")
        q = quaternion_from_euler(math.radians(r), math.radians(p), math.radians(yaw))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.arm_group.set_pose_target(pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        return success

    def _smart_move_to_pose(self, x, y, z, r, p, yaw, auto_fix=True):
        # 1. Intento directo
        if self._move_to_pose(x, y, z, r, p, yaw): return True
        if not auto_fix: return False

        self._feedback("Direct plan failed. Trying Smart Rotation...")
        
        # 2. Barrido inteligente
        offsets = []
        for i in range(15, 181, 15):
            offsets.append(i)
            offsets.append(-i)
            
        for offset in offsets:
            if self.check_stop(): return False
            test_yaw = yaw + offset
            # Normalizar
            if test_yaw > 180: test_yaw -= 360
            if test_yaw < -180: test_yaw += 360
            
            if self._move_to_pose(x, y, z, r, p, test_yaw):
                self._feedback(f"Success with Yaw={test_yaw}")
                return True
        return False

    def _move_joints(self, joints):
        if self.check_stop(): return False
        joint_names = self.arm_group.get_active_joints()
        if len(joints) != len(joint_names): return False
        self.arm_group.set_joint_value_target(dict(zip(joint_names, joints)))
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        return success

    def _move_gripper(self, state):
        if self.check_stop(): return False
        vals = [0.044, 0.044] if state == "open" else [0.005, 0.005]
        self.gripper_group.set_joint_value_target(vals)
        return self.gripper_group.go(wait=True)

    # --- GUI HELPERS ---
    def gui_go_home(self):
        # Pose "Home" segura manual
        HOME_JOINTS = [0.20, -0.50, -0.20, 1.50, -1.00, 1.00, 0.00]
        self.stop_flag = False
        self.is_busy = True
        self.log_signal.emit("Manual: Going Home...")
        res = self._move_joints(HOME_JOINTS)
        self.is_busy = False
        return res, "Home Reached" if res else "Failed"

    def gui_go_ready(self):
        self.stop_flag = False
        self.is_busy = True
        self.log_signal.emit("Manual: Going Ready...")
        res = self._move_joints(READY_JOINTS)
        self.is_busy = False
        return res, "Ready Reached" if res else "Failed"

    def gui_go_rest(self):
        # Si tienes REST_JOINTS definidos arriba, úsalos
        if 'REST_JOINTS' not in globals(): return False, "No REST pose defined"
        self.stop_flag = False
        self.is_busy = True
        self.log_signal.emit("Manual: Going Rest...")
        res = self._move_joints(REST_JOINTS)
        self.is_busy = False
        return res, "Rest Reached" if res else "Failed"

    # --- ACTION CALLBACK ---
    def execute_action_cb(self, goal):
        if self.is_busy:
            self.server.set_aborted(PickUpResult(False, "Robot Busy"))
            return

        self.is_busy = True
        self.stop_flag = False
        result = PickUpResult()
        
        try:
            self._feedback("Starting Sequence...")
            
            # 1. Ready
            if not self._move_joints(READY_JOINTS): raise Exception("Ready Fail")
            self._move_gripper("open")

            # 2. Pick
            r, p, yaw = DEFAULT_ROT['r'], DEFAULT_ROT['p'], DEFAULT_ROT['y']
            auto = False
            
            self._feedback("Picking...")
            # Aprox (+15cm)
            if not self._smart_move_to_pose(goal.pick_x, goal.pick_y, goal.pick_z + 0.15, r, p, yaw, auto):
                 raise Exception("Pick App Fail")
            # Down
            if not self._smart_move_to_pose(goal.pick_x, goal.pick_y, goal.pick_z, r, p, yaw, auto):
                 raise Exception("Pick Down Fail")
            
            self._move_gripper("close")
            rospy.sleep(0.5)
            
            # Up
            if not self._smart_move_to_pose(goal.pick_x, goal.pick_y, goal.pick_z + 0.15, r, p, yaw, auto):
                 raise Exception("Pick Up Fail")

            # 3. Place
            self._feedback("Placing...")
            # Aprox
            if not self._smart_move_to_pose(goal.place_x, goal.place_y, goal.place_z + 0.15, r, p, yaw, auto):
                 raise Exception("Place App Fail")
            # Down
            if not self._smart_move_to_pose(goal.place_x, goal.place_y, goal.place_z, r, p, yaw, auto):
                 raise Exception("Place Down Fail")
            
            self._move_gripper("open")
            rospy.sleep(0.5)
            
            # Up
            if not self._smart_move_to_pose(goal.place_x, goal.place_y, goal.place_z + 0.15, r, p, yaw, auto):
                 raise Exception("Place Up Fail")

            # 4. Return
            self._move_joints(READY_JOINTS)
            
            result.success = True
            result.message = "OK"
            self.server.set_succeeded(result)
            self.log_signal.emit("Action Finished.")

        except Exception as e:
            if not self.stop_flag:
                self.server.set_aborted(PickUpResult(False, str(e)))
            self.log_signal.emit(f"Sequence Error: {e}")
        
        finally:
            self.is_busy = False

# --- GUI MINIMALISTA ---
class MinimalGUI(QtWidgets.QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()
        self.controller.log_signal.connect(self.log)

    def init_ui(self):
        self.setWindowTitle("Pick & Place Server")
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: #222; color: #eee;")

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("ROBOT SERVER (NO TORSO)")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        layout.addWidget(title)

        btn_home = QtWidgets.QPushButton("GO HOME")
        btn_home.setStyleSheet("background-color: #4CAF50; padding: 10px; font-weight: bold;")
        btn_home.clicked.connect(self.on_home)
        layout.addWidget(btn_home)

        btn_ready = QtWidgets.QPushButton("GO READY")
        btn_ready.setStyleSheet("background-color: #2196F3; padding: 10px; font-weight: bold;")
        btn_ready.clicked.connect(self.on_ready)
        layout.addWidget(btn_ready)

        # Si definiste REST_JOINTS, descomenta esto para mostrar el botón
        # btn_rest = QtWidgets.QPushButton("GO REST")
        # btn_rest.setStyleSheet("background-color: #8E44AD; padding: 10px; font-weight: bold;")
        # btn_rest.clicked.connect(self.on_rest)
        # layout.addWidget(btn_rest)

        btn_stop = QtWidgets.QPushButton("EMERGENCY STOP")
        btn_stop.setStyleSheet("background-color: #f44336; padding: 15px; font-weight: bold;")
        btn_stop.clicked.connect(self.on_stop)
        layout.addWidget(btn_stop)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #111; color: #0f0; font-family: monospace;")
        layout.addWidget(self.log_box)

    def log(self, msg):
        self.log_box.append(f"> {msg}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def on_home(self):
        self.worker = WorkerThread(self.controller.gui_go_home)
        self.worker.finished_signal.connect(lambda s, m: self.log(f"Home: {m}"))
        self.worker.start()

    def on_ready(self):
        self.worker = WorkerThread(self.controller.gui_go_ready)
        self.worker.finished_signal.connect(lambda s, m: self.log(f"Ready: {m}"))
        self.worker.start()

    def on_rest(self):
        self.worker = WorkerThread(self.controller.gui_go_rest)
        self.worker.finished_signal.connect(lambda s, m: self.log(f"Rest: {m}"))
        self.worker.start()

    def on_stop(self):
        self.controller.stop_robot()

def main():
    rospy.init_node('pickup_server_gui', anonymous=False)
    app = QtWidgets.QApplication(sys.argv)
    controller = RobotController()
    gui = MinimalGUI(controller)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()