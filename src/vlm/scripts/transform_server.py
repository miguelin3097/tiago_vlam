#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from vlm.srv import TransformPose, TransformPoseResponse

class TransformServer:
    def __init__(self):
        rospy.init_node('transform_pose_server')
        
        # 1. Herramientas TF y CV
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()
        
        # Variables de estado
        self.cam_info_received = False
        self.latest_depth = None

        # 2. Suscriptores (Igual que el GUI)
        # Usamos depth_registered para asegurar alineación
        rospy.Subscriber('/xtion/depth_registered/camera_info', CameraInfo, self.info_cb)
        rospy.Subscriber('/xtion/depth_registered/image_raw', Image, self.depth_cb)

        # 3. Iniciar Servicio
        self.server = rospy.Service('/vlm/transform_pose', TransformPose, self.handle_transform)
        rospy.loginfo("Transform Server READY. Waiting for Depth Images...")

    def info_cb(self, msg):
        if not self.cam_info_received:
            self.cam_model.fromCameraInfo(msg)
            self.cam_info_received = True
            rospy.loginfo("Camera Info Received.")

    def depth_cb(self, msg):
        try:
            # Guardamos la imagen de profundidad más reciente (igual que el GUI)
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr(f"Error reading depth image: {e}")

    def handle_transform(self, req):
        resp = TransformPoseResponse()
        resp.success = False

        # 1. Validaciones básicas
        if not self.cam_info_received:
            rospy.logwarn("Waiting for camera info...")
            return resp
        
        if self.latest_depth is None:
            rospy.logwarn("No depth image received yet.")
            return resp

        # 2. Obtener dimensiones y validar límites (Lógica del GUI)
        h, w = self.latest_depth.shape
        u, v = int(req.cam_x), int(req.cam_y) # Asumimos que llegan en píxeles

        if u < 0 or u >= w or v < 0 or v >= h:
            rospy.logerr(f"Pixel ({u},{v}) is out of bounds ({w}x{h})")
            return resp

        # 3. Leer profundidad real del sensor
        depth = self.latest_depth[v, u]
        
        # Filtro de profundidad (Nan o <= 0.05)
        if np.isnan(depth) or depth <= 0.05:
            rospy.logwarn(f"Invalid depth at ({u},{v}): {depth}")
            return resp

        # 4. Proyectar a 3D (Marco Óptico)
        ray_3d = self.cam_model.projectPixelTo3dRay((u, v))
        
        p_cam = PointStamped()
        p_cam.header.frame_id = self.cam_model.tfFrame()
        p_cam.header.stamp = rospy.Time(0)
        
        # Aplicar la profundidad real leída del sensor
        p_cam.point.x = ray_3d[0] * depth
        p_cam.point.y = ray_3d[1] * depth
        p_cam.point.z = ray_3d[2] * depth

        try:
            # 5. Transformar a base_footprint
            transform = self.tf_buffer.lookup_transform("base_footprint", 
                                                      p_cam.header.frame_id, 
                                                      rospy.Time(0), 
                                                      rospy.Duration(1.0))
            
            p_base = tf2_geometry_msgs.do_transform_point(p_cam, transform)

            # ---------------------------------------------------------
            # IMPORTANTE: APLICANDO EL MISMO OFFSET QUE EL GUI
            # ---------------------------------------------------------
            p_base.point.y -= 0.19 
            # ---------------------------------------------------------

            resp.base_x = p_base.point.x
            resp.base_y = p_base.point.y
            resp.base_z = p_base.point.z
            resp.success = True
            
            rospy.loginfo(f"Transformed: px({u},{v}) d={depth:.2f}m -> base({resp.base_x:.3f}, {resp.base_y:.3f}, {resp.base_z:.3f})")

        except Exception as e:
            rospy.logerr(f"TF Error: {e}")
            resp.success = False

        return resp

if __name__ == "__main__":
    try:
        TransformServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass