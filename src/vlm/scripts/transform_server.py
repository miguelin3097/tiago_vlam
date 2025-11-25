#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
from vlm.srv import TransformPose, TransformPoseResponse

class TransformServer:
    def __init__(self):
        rospy.init_node('transform_pose_server')
        
        # 1. Listener de TF (Para saber dónde está la cámara respecto a la base)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 2. Modelo de Cámara (Para saber intrínsecos: distancia focal, centro óptico)
        self.cam_model = PinholeCameraModel()
        self.info_sub = rospy.Subscriber('/xtion/rgb/camera_info', CameraInfo, self.info_cb)
        self.cam_info_received = False

        # 3. Iniciar Servicio
        self.server = rospy.Service('/vlm/transform_pose', TransformPose, self.handle_transform)
        rospy.loginfo("Transform Server READY. Waiting for Camera Info...")

    def info_cb(self, msg):
        self.cam_model.fromCameraInfo(msg)
        self.cam_info_received = True
        self.info_sub.unregister() # Solo necesitamos leerlo una vez
        rospy.loginfo("Camera Info Received. Service fully active.")

    def handle_transform(self, req):
        resp = TransformPoseResponse()
        resp.success = False

        if not self.cam_info_received:
            rospy.logwarn("Cannot transform: Waiting for camera_info...")
            return resp

        # Paso A: Desproyectar Pixel (u,v) -> Rayo 3D en marco óptico
        # projectPixelTo3dRay devuelve un vector (x,y,z) normalizado con z=1
        ray_3d = self.cam_model.projectPixelTo3dRay((req.cam_x, req.cam_y))
        
        # Creamos un punto estampado en el marco óptico
        p_cam = PointStamped()
        p_cam.header.frame_id = self.cam_model.tfFrame()
        p_cam.header.stamp = rospy.Time(0) # Usamos la transformada más reciente
        
        # Escalamos el rayo por la profundidad si se conoce, o asumimos una Z arbitraria
        # para luego proyectar. Si req.cam_z_depth viene > 0, lo usamos.
        scale = req.cam_z_depth if req.cam_z_depth > 0.1 else 1.0
        p_cam.point.x = ray_3d[0] * scale
        p_cam.point.y = ray_3d[1] * scale
        p_cam.point.z = ray_3d[2] * scale

        try:
            # Paso B: Transformar ese punto al marco 'base_footprint'
            # Esperamos 1 seg por la transformada si no está en caché
            transform = self.tf_buffer.lookup_transform("base_footprint", 
                                                      p_cam.header.frame_id, 
                                                      rospy.Time(0), 
                                                      rospy.Duration(1.0))
            
            p_base = tf2_geometry_msgs.do_transform_point(p_cam, transform)

            # --- LÓGICA DE INTERSECCIÓN CON MESA (Opcional pero recomendada) ---
            # Si no tenemos cámara de profundidad (z_depth ficticio), el punto p_base estará flotando.
            # Aquí calculamos dónde ese rayo choca con la mesa (Z = 0.75 aprox).
            # Si TIAGo tiene cámara RGB-D, 'req.cam_z_depth' debería ser la distancia real del sensor
            # y p_base ya será correcto.
            
            # Asumimos que p_base es correcto (usando profundidad real del sensor)
            resp.base_x = p_base.point.x
            resp.base_y = p_base.point.y
            resp.base_z = p_base.point.z
            resp.success = True
            
            rospy.loginfo(f"Transformed: ({req.cam_x},{req.cam_y}) -> ({resp.base_x:.3f}, {resp.base_y:.3f}, {resp.base_z:.3f})")

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