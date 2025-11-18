#!/usr/bin/env python3
import sys
import rospy
import math
import moveit_commander
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QGroupBox, QMessageBox, QTextEdit, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QDoubleValidator, QFont

# --- CONFIGURACIÓN ---
READY_POS = {'x': 0.55, 'y': -0.1, 'z': 1.0}
READY_ROT = {'r': 90.0, 'p': 90.0, 'y': 0.0}

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
            self.finished_signal.emit(False, f"Error crítico: {str(e)}")

# --- CONTROLADOR (BACKEND) ---
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
        
        self.arm_group.set_planning_time(5.0)
        self.arm_group.set_num_planning_attempts(10)
        self.arm_group.set_goal_position_tolerance(0.005)
        self.arm_group.set_goal_orientation_tolerance(0.05)

    # --- UTILIDADES ---
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
        self.arm_group.set_pose_target(pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        return success

    def _move_gripper(self, state):
        # TIAGo Gripper Limits aprox: 0.002 (cerrado) a 0.045 (abierto)
        
        if state == "open":
            # Abrimos a 4.4cm (un poco menos del maximo 4.5 para seguridad)
            vals = [0.044, 0.044] 
        else:
            # CERRAR: Cambiamos 0.00 por 0.005 (5 milímetros)
            # Esto evita el error "Target out of bounds"
            vals = [0.005, 0.005]
            
        self.gripper_group.set_joint_value_target(vals)
        return self.gripper_group.go(wait=True)

    # --- FUNCIONES DE TAREA ---
    def stop_robot(self, logger):
        self.arm_group.stop()
        self.gripper_group.stop()
        return True, "Parada ejecutada."

    def go_home_safe(self, logger):
        logger.emit("Moviendo a Home Safe...")
        joint_names = self.arm_group.get_active_joints()
        safe_values = [0.20, -0.50, -0.20, 1.50, -1.00, 1.00, 0.00]
        if len(joint_names) != len(safe_values): return False, "Error joints."
        
        self.arm_group.set_joint_value_target(dict(zip(joint_names, safe_values)))
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        return success, "Robot en HOME Safe."

    def go_to_ready_pose(self, logger):
        logger.emit("Yendo a posición READY...")
        pose = self._create_pose(READY_POS['x'], READY_POS['y'], READY_POS['z'],
                                 READY_ROT['r'], READY_ROT['p'], READY_ROT['y'])
        return self._move_to_pose(pose), "Robot en READY."

    def execute_manual_move(self, logger, x, y, z, r, p, yw):
        """Movimiento manual simple"""
        logger.emit(f"Moviendo Manual a: ({x}, {y}, {z}) RPY:({r}, {p}, {yw})")
        pose = self._create_pose(x, y, z, r, p, yw)
        success = self._move_to_pose(pose)
        return success, "Movimiento manual finalizado."

    def execute_gripper(self, logger, state):
        logger.emit(f"Gripper: {state}")
        return self._move_gripper(state), f"Gripper {state}"

    def execute_pick_place(self, logger, pick_x, pick_y, pick_z, place_x, place_y, place_z):
        R, P, Y = READY_ROT['r'], READY_ROT['p'], READY_ROT['y']
        OFFSET_Z = 0.15 

        logger.emit("1. Inicio secuencia Pick & Place...")
        pose_ready = self._create_pose(READY_POS['x'], READY_POS['y'], READY_POS['z'], R, P, Y)
        if not self._move_to_pose(pose_ready): return False, "Fallo inicio."
        
        self._move_gripper("open")

        # PICK
        logger.emit(f"2. Pick en ({pick_x}, {pick_y})...")
        pose_pick_app = self._create_pose(pick_x, pick_y, pick_z + OFFSET_Z, R, P, Y)
        if not self._move_to_pose(pose_pick_app): return False, "Fallo aprox Pick."

        pose_pick = self._create_pose(pick_x, pick_y, pick_z, R, P, Y)
        if not self._move_to_pose(pose_pick): return False, "Fallo bajar Pick."

        logger.emit("   > Cerrando pinza...")
        self._move_gripper("close")
        rospy.sleep(0.5)

        logger.emit("   > Subiendo...")
        if not self._move_to_pose(pose_pick_app): return False, "Fallo subir Pick."

        # PLACE
        logger.emit(f"3. Place en ({place_x}, {place_y})...")
        pose_place_app = self._create_pose(place_x, place_y, place_z + OFFSET_Z, R, P, Y)
        if not self._move_to_pose(pose_place_app): return False, "Fallo aprox Place."

        pose_place = self._create_pose(place_x, place_y, place_z, R, P, Y)
        if not self._move_to_pose(pose_place): return False, "Fallo bajar Place."

        logger.emit("   > Abriendo pinza...")
        self._move_gripper("open")
        rospy.sleep(0.5)

        logger.emit("   > Subiendo...")
        if not self._move_to_pose(pose_place_app): return False, "Fallo subir Place."

        logger.emit("4. Volviendo a Inicio...")
        self._move_to_pose(pose_ready)
        return True, "Secuencia terminada."

# --- INTERFAZ GRÁFICA ---
class TiagoControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = RobotController()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TIAGo Master Control")
        self.setGeometry(100, 100, 550, 850) # Ventana más alta
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-family: Arial;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # --- Título ---
        title = QLabel("TIAGo Command Center")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #00d0ff; margin: 5px;")
        main_layout.addWidget(title)

        # --- Botones HOME / READY / GRIPPER ---
        top_layout = QHBoxLayout()
        
        btn_home = QPushButton("HOME")
        btn_home.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 8px;")
        btn_home.clicked.connect(self.action_go_home)
        
        btn_ready = QPushButton("READY")
        btn_ready.setStyleSheet("background-color: #2196F3; font-weight: bold; padding: 8px;")
        btn_ready.clicked.connect(self.action_go_ready)

        btn_g_open = QPushButton("ABRIR PINZA")
        btn_g_open.setStyleSheet("background-color: #555; border: 1px solid #777;")
        btn_g_open.clicked.connect(lambda: self.action_gripper("open"))

        btn_g_close = QPushButton("CERRAR PINZA")
        btn_g_close.setStyleSheet("background-color: #555; border: 1px solid #777;")
        btn_g_close.clicked.connect(lambda: self.action_gripper("close"))

        top_layout.addWidget(btn_home)
        top_layout.addWidget(btn_ready)
        top_layout.addWidget(btn_g_open)
        top_layout.addWidget(btn_g_close)
        main_layout.addLayout(top_layout)

        # --- SECCIÓN 1: CONTROL MANUAL ---
        self.create_manual_section(main_layout)

        # --- SECCIÓN 2: PICK & PLACE AUTO ---
        self.create_pick_place_section(main_layout)

        # --- STOP ---
        # CORRECCIÓN: Guardamos el botón en self.btn_stop para acceder a él fácilmente
        self.btn_stop = QPushButton("PARADA DE EMERGENCIA")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 15px; font-size: 12pt; margin-top: 10px;")
        self.btn_stop.clicked.connect(self.action_stop)
        main_layout.addWidget(self.btn_stop)

        # --- LOG ---
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #101010; color: #00ff00; font-family: Monospace;")
        self.log_box.setFixedHeight(120)
        main_layout.addWidget(self.log_box)

        self.status_label = QLabel("Listo.")
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

    def create_manual_section(self, parent):
        group = QGroupBox("Control Manual (Pose + Rotación)")
        group.setStyleSheet("QGroupBox { border: 1px solid #777; border-radius: 5px; margin-top: 10px; font-weight: bold; }")
        layout = QVBoxLayout()

        # Fila 1: X Y Z
        r1 = QHBoxLayout()
        self.man_x = self.create_input("X:", "0.55", r1)
        self.man_y = self.create_input("Y:", "-0.10", r1)
        self.man_z = self.create_input("Z:", "1.00", r1)
        layout.addLayout(r1)

        # Fila 2: R P Y
        r2 = QHBoxLayout()
        self.man_r = self.create_input("Roll:", "90.0", r2)
        self.man_p = self.create_input("Pitch:", "90.0", r2)
        self.man_yw = self.create_input("Yaw:", "0.0", r2)
        layout.addLayout(r2)

        btn_move = QPushButton("MOVER A POSE")
        btn_move.setStyleSheet("background-color: #FF9800; color: black; font-weight: bold; padding: 8px;")
        btn_move.clicked.connect(self.action_manual_move)
        layout.addWidget(btn_move)

        group.setLayout(layout)
        parent.addWidget(group)

    def create_pick_place_section(self, parent):
        group = QGroupBox("Secuencia Automática Pick & Place")
        group.setStyleSheet("QGroupBox { border: 1px solid #00d0ff; border-radius: 5px; margin-top: 10px; font-weight: bold; }")
        layout = QVBoxLayout()
        
        lbl = QLabel("Usa Rotación 'Ready' por defecto")
        lbl.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(lbl)

        # Inputs
        f_pick = QFrame()
        f_pick.setStyleSheet("background-color: #333; border-radius: 5px;")
        lp = QHBoxLayout(f_pick)
        lp.addWidget(QLabel("PICK:"))
        self.pp_px = self.create_input("X", "0.60", lp)
        self.pp_py = self.create_input("Y", "-0.20", lp)
        self.pp_pz = self.create_input("Z", "0.75", lp)
        layout.addWidget(f_pick)

        f_place = QFrame()
        f_place.setStyleSheet("background-color: #333; border-radius: 5px;")
        ll = QHBoxLayout(f_place)
        ll.addWidget(QLabel("PLACE:"))
        self.pp_lx = self.create_input("X", "0.60", ll)
        self.pp_ly = self.create_input("Y", "0.20", ll)
        self.pp_lz = self.create_input("Z", "0.75", ll)
        layout.addWidget(f_place)

        btn_start = QPushButton("INICIAR SECUENCIA")
        btn_start.setStyleSheet("background-color: #00BCD4; color: black; font-weight: bold; padding: 10px;")
        btn_start.clicked.connect(self.action_start_sequence)
        layout.addWidget(btn_start)

        group.setLayout(layout)
        parent.addWidget(group)

    def create_input(self, label, default, layout):
        l = QLabel(label)
        i = QLineEdit(default)
        i.setFixedWidth(50)
        i.setStyleSheet("background-color: #555; color: white; border: none; padding: 3px;")
        layout.addWidget(l)
        layout.addWidget(i)
        return i

    # --- LÓGICA ---
    def log(self, msg):
        self.log_box.append(f"> {msg}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_busy(self, busy):
        self.centralWidget().setEnabled(not busy)
        # CORRECCIÓN AQUI: Usamos la variable guardada, no findChild
        if busy:
            self.btn_stop.setEnabled(True) 

    def on_finished(self, success, msg):
        self.set_busy(False)
        color = "#00ff00" if success else "#ff0000"
        self.status_label.setStyleSheet(f"color: {color}")
        self.status_label.setText(msg)
        self.log(f"FIN: {msg}")

    # --- ACCIONES ---
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
        except ValueError:
            QMessageBox.warning(self, "Error", "Números inválidos.")
            return
        
        self.set_busy(True)
        self.worker = WorkerThread(self.controller.execute_manual_move, x, y, z, r, p, yw)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def action_start_sequence(self):
        try:
            px, py, pz = float(self.pp_px.text()), float(self.pp_py.text()), float(self.pp_pz.text())
            lx, ly, lz = float(self.pp_lx.text()), float(self.pp_ly.text()), float(self.pp_lz.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Números inválidos.")
            return

        self.set_busy(True)
        self.worker = WorkerThread(self.controller.execute_pick_place, px, py, pz, lx, ly, lz)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def action_stop(self):
        self.log("!!! STOP !!!")
        self.controller.stop_robot(None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TiagoControlWindow()
    win.show()
    sys.exit(app.exec_())