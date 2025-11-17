#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped

def open_right_gripper(robot, gripper_group):
    """
    Abre la pinza derecha usando la misma lógica que antes:
    cogemos la primera articulación del grupo y la llevamos casi al límite superior.
    """
    active_joints = gripper_group.get_active_joints()
    rospy.loginfo("Articulaciones activas del gripper_right: %s", active_joints)

    main_joint_name = active_joints[0]
    joint_obj = robot.get_joint(main_joint_name)
    lower, upper = joint_obj.bounds()
    rospy.loginfo("Límites de %s: lower=%.6f, upper=%.6f", main_joint_name, lower, upper)

    # Abrimos casi al máximo
    eps = 1e-3
    open_val = upper - 2 * eps

    rospy.loginfo("Abriendo pinza: %s -> %.6f", main_joint_name, open_val)
    target_dict = {main_joint_name: open_val}
    gripper_group.set_joint_value_target(target_dict)

    success = gripper_group.go(wait=True)
    gripper_group.stop()
    rospy.loginfo("Resultado de apertura de pinza: %s", success)


def move_arm_to_table_pose(arm_group):
    """
    Mueve el brazo derecho a una pose 'sobre la mesa' frente al robot.
    Ajusta las coordenadas x, y, z según dónde esté tu mesa/objeto.
    """
    planning_frame = arm_group.get_planning_frame()
    ee_link = arm_group.get_end_effector_link()
    rospy.loginfo("Planning frame: %s", planning_frame)
    rospy.loginfo("End-effector link: %s", ee_link)

    # Leemos la pose actual para reutilizar la orientación
    current_pose = arm_group.get_current_pose(ee_link).pose
    rospy.loginfo("Pose actual del end-effector: %s", current_pose)

    target_pose = PoseStamped()
    target_pose.header.frame_id = planning_frame  # normalmente "base_footprint"
    target_pose.header.stamp = rospy.Time.now()

    # 🔧 AJUSTA ESTOS VALORES A TU MESA / OBJETO 🔧
    target_pose.pose.position.x = 0.6   # hacia delante
    target_pose.pose.position.y = -0.2  # hacia la derecha (para brazo derecho)
    target_pose.pose.position.z = 0.75  # altura aproximada de la mesa

    # Usamos la MISMA orientación que la actual para no complicarnos con cuaterniones
    target_pose.pose.orientation = current_pose.orientation

    rospy.loginfo("Nueva pose objetivo del end-effector: %s", target_pose.pose)

    arm_group.set_pose_target(target_pose, ee_link)

    rospy.loginfo("Planificando y ejecutando movimiento del brazo...")
    success = arm_group.go(wait=True)
    arm_group.stop()
    arm_group.clear_pose_targets()

    rospy.loginfo("Resultado de movimiento de brazo: %s", success)


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('tiago_arm_to_table_demo', anonymous=True)

    robot = moveit_commander.RobotCommander()
    gripper_group = moveit_commander.MoveGroupCommander("gripper_right")
    arm_group = moveit_commander.MoveGroupCommander("arm_right")

    rospy.loginfo("Esperando 2 segundos antes de mover...")
    rospy.sleep(2.0)

    # 1) Abrimos la pinza
    open_right_gripper(robot, gripper_group)

    # 2) Movemos el brazo a una pose sobre la mesa
    move_arm_to_table_pose(arm_group)

    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
