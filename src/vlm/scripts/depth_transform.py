#!/usr/bin/env python3
import sys
import json
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

# PyQt5 Imports
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QPushButton, QLabel, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

class TiagoVisionBackend:
    """
    Backend class handling ROS logic:
    - Subscribes to camera topics
    - Calculates 3D transformations
    """
    def __init__(self):
        # Initialize node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node('tiago_gui_calculator', anonymous=True)

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Camera Tools
        self.camera_model = PinholeCameraModel()
        self.bridge = CvBridge()
        self.latest_depth = None
        self.cam_info_ok = False
        
        # Subscribers (Standard TIAGo topics)
        # Using depth_registered to ensure pixel alignment with RGB
        rospy.Subscriber("/xtion/depth_registered/camera_info", CameraInfo, self.info_cb)
        rospy.Subscriber("/xtion/depth_registered/image_raw", Image, self.depth_cb)

    def info_cb(self, msg):
        if not self.cam_info_ok:
            self.camera_model.fromCameraInfo(msg)
            self.cam_info_ok = True
    
    def depth_cb(self, msg):
        try:
            # Convert ROS message to OpenCV image (passthrough keeps original encoding)
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

    def get_robot_coordinates(self, u, v):
        """
        Converts pixel (u, v) -> Point(x, y, z) in 'base_footprint' frame
        INCLUYE OFFSET MANUAL DE -19 CM EN EJE Y.
        """
        if not self.cam_info_ok or self.latest_depth is None:
            return "No Data", None

        # Check bounds
        h, w = self.latest_depth.shape
        if u < 0 or u >= w or v < 0 or v >= h:
            return "Out of Bounds", None

        # 1. Get Depth
        depth = self.latest_depth[v, u]
        
        # Filter invalid depth readings
        if np.isnan(depth) or depth <= 0.05: 
            return "Nan/Zero Depth", None

        # 2. Project to 3D (Camera Optical Frame)
        ray = self.camera_model.projectPixelTo3dRay((u, v))
        p_cam = PointStamped()
        p_cam.header.frame_id = self.camera_model.tfFrame()
        p_cam.header.stamp = rospy.Time(0)
        
        p_cam.point.x = ray[0] * depth
        p_cam.point.y = ray[1] * depth
        p_cam.point.z = ray[2] * depth

        # 3. Transform to Robot Base
        try:
            transform = self.tf_buffer.lookup_transform("base_footprint", 
                                                        p_cam.header.frame_id, 
                                                        rospy.Time(0), 
                                                        rospy.Duration(0.5))
            
            p_base = tf2_geometry_msgs.do_transform_point(p_cam, transform)
            
            # ---------------------------------------------------------
            # APLICANDO EL OFFSET MANUAL DE -19 CM AL EJE Y
            # ---------------------------------------------------------
            p_base.point.y -= 0.19 
            # ---------------------------------------------------------

            return "OK", p_base.point
        except Exception as e:
            return f"TF Error: {str(e)}", None

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.backend = TiagoVisionBackend()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TIAGo JSON to Coordinates")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("font-size: 14px;")

        layout = QVBoxLayout()

        # --- SECTION 1: JSON Input ---
        lbl_instr = QLabel("1. Paste the JSON returned by the model here:")
        lbl_instr.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(lbl_instr)

        self.txt_input = QTextEdit()
        self.txt_input.setPlaceholderText('{\n  "cubes": [...],\n  ...\n}')
        layout.addWidget(self.txt_input)

        # --- SECTION 2: Button ---
        self.btn_calc = QPushButton(" Calculate Robot Coordinates (base_footprint) ")
        self.btn_calc.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_calc.clicked.connect(self.process_json)
        layout.addWidget(self.btn_calc)

        # --- SECTION 3: Results ---
        lbl_res = QLabel("2. Calculated Coordinates:")
        lbl_res.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(lbl_res)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Color", "Pixel (u, v)", "X (Front)", "Y (Left/Right)", "Z (Height)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Status Label
        self.lbl_status = QLabel("Status: Waiting for input...")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.lbl_status)

        self.setLayout(layout)

    def process_json(self):
        # 1. Verify Robot Connection
        if self.backend.latest_depth is None:
            QMessageBox.warning(self, "Connection Error", 
                                "Not receiving images from TIAGo.\nCheck ROS connection (/xtion/depth_registered/image_raw).")
            return

        json_text = self.txt_input.toPlainText()
        if not json_text:
            return

        try:
            data = json.loads(json_text)
            cubes = data.get("cubes", [])
            
            # Get real image dimensions from backend
            img_h, img_w = self.backend.latest_depth.shape
            
            self.table.setRowCount(0)
            
            for i, cube in enumerate(cubes):
                color = cube.get("color", "unknown")
                yx = cube.get("yx", [])
                
                # Validate data
                if not isinstance(yx, list) or len(yx) != 2:
                    continue

                # --- DENORMALIZATION LOGIC ---
                # JSON comes as [y, x] (normalized 0-1000)
                y_norm = yx[0]
                x_norm = yx[1]
                
                # Convert to real pixels (u=horizontal=x, v=vertical=y)
                # Using (size - 1) to stay within array indices
                u_pixel = int(round((float(x_norm) / 1000.0) * (img_w - 1)))
                v_pixel = int(round((float(y_norm) / 1000.0) * (img_h - 1)))
                
                # Clamp values to stay inside image
                u_pixel = max(0, min(img_w - 1, u_pixel))
                v_pixel = max(0, min(img_h - 1, v_pixel))

                # --- GET 3D COORDINATES ---
                status, point_3d = self.backend.get_robot_coordinates(u_pixel, v_pixel)

                # Fill Table
                self.table.insertRow(i)
                self.table.setItem(i, 0, QTableWidgetItem(color))
                self.table.setItem(i, 1, QTableWidgetItem(f"{u_pixel}, {v_pixel}"))

                if point_3d:
                    # Format to 3 decimal places (millimeters)
                    x_str = f"{point_3d.x:.3f}"
                    y_str = f"{point_3d.y:.3f}"
                    z_str = f"{point_3d.z:.3f}"
                    
                    # Green background for valid coordinates
                    item_x = QTableWidgetItem(x_str)
                    item_x.setBackground(Qt.green) # type: ignore
                    self.table.setItem(i, 2, item_x)
                    self.table.setItem(i, 3, QTableWidgetItem(y_str))
                    self.table.setItem(i, 4, QTableWidgetItem(z_str))
                else:
                    # Show error status in red text if failed
                    item_err = QTableWidgetItem(status)
                    item_err.setForeground(Qt.red) # type: ignore
                    self.table.setItem(i, 2, item_err)
            
            self.lbl_status.setText(f"Processed {len(cubes)} objects. Detected resolution: {img_w}x{img_h}")

        except json.JSONDecodeError:
            QMessageBox.critical(self, "JSON Error", "Input text is not a valid JSON.")
        except Exception as e:
            QMessageBox.critical(self, "Fatal Error", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())