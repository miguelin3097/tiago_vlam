# env_loader.py
import os
import rospkg
import rospy

def load_env_file(package_name, filename=".env"):
    """
    Carga variables de entorno desde un archivo .env dentro de un paquete ROS.
    No necesita librerías externas.
    """
    rp = rospkg.RosPack()
    pkg_path = rp.get_path(package_name)
    env_path = os.path.join(pkg_path, filename)

    if not os.path.exists(env_path):
        rospy.logwarn("Archivo .env no encontrado en: %s", env_path)
        return False

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

    return True
