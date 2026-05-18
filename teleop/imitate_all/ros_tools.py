from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import rospy


class Lister(object):

    @staticmethod
    def pose_to_list(no_position, pose: Pose) -> list:
        lis = []
        if not no_position:
            lis += [pose.position.x, pose.position.y, pose.position.z]
        ori = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        if sum(ori) > 0:
            lis += ori
        return lis

    @staticmethod
    def joint_state_to_list(joint_state: JointState) -> list:
        """Convert all fields of JointState to list"""
        return (
            list(joint_state.position)
            + list(joint_state.velocity)
            + list(joint_state.effort)
        )

    @staticmethod
    def data_field_to_list(data) -> list:
        data = data.data
        if not isinstance(data, (tuple, list)):
            data = [data]
        return data

    @staticmethod
    def list_to_pose(data: list) -> Pose:
        """Auto convert based on the length of the list"""
        pose = Pose()
        if len(data) >= 3:
            pose.position.x = data[0]
            pose.position.y = data[1]
            pose.position.z = data[2]
        if len(data) >= 4:
            pose.orientation.x = data[3]
            pose.orientation.y = data[4]
            pose.orientation.z = data[5]
            pose.orientation.w = data[6]
        return pose

    @staticmethod
    def list_to_joint_state(fields: str, data: list) -> JointState:
        """Convert list to JointState
        Parameters:
        data: list
            The list to convert
        fields: str
            The fields to convert. Default is 'p' which means only position
            'p' for position
            'v' for velocity
            'e' for effort
            'pv' for position and velocity
            'pe' for position and effort
            've' for velocity and effort
            'a' or 'pve' for position, velocity and effort
        """
        joint_state = JointState()
        if fields == "a":
            fields = "pve"
        dim = len(data) / len(fields)
        assert dim % 1 == 0, "All fields must have the same dim"
        start = 0
        end = int(dim)
        if "p" in fields:
            joint_state.position = data[start:end]
            start = end
            end = int(start + dim)
        if "v" in fields:
            joint_state.velocity = data[start:end]
            start = end
            end = int(start + dim)
        if "e" in fields:
            joint_state.effort = data[start:end]
        joint_state.header.stamp = rospy.Time.now()
        return joint_state

    @staticmethod
    def list_to_given_data_field(type: str, data: list):
        if len(data) == 1:
            data = data[0]
        return type(data=data)


if __name__ == "__main__":
    # all to list
    pose = Pose()
    pose.position.x = 1
    pose.position.y = 2
    pose.position.z = 3
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 0
    pose_list_1 = Lister.pose_to_list(pose)
    pose.orientation.w = 1
    pose_list_2 = Lister.pose_to_list(pose)

    joint_state = JointState()
    joint_state.position = [1, 2, 3]
    joint_state.velocity = [4, 5, 6]
    joint_state.effort = [7, 8, 9]
    joint_state_list_1 = Lister.joint_state_to_list(joint_state)
    joint_state.velocity = []
    joint_state_list_2 = Lister.joint_state_to_list(joint_state)

    from std_msgs.msg import (
        Float32MultiArray,
        Int32,
    )

    float32_multi_array = Float32MultiArray()
    float32_multi_array.data = [1, 2, 3]
    float32_multi_array_list_1 = Lister.data_field_to_list(float32_multi_array)

    int32 = Int32()
    int32.data = 1
    int32_list = Lister.data_field_to_list(int32)

    # list to all
    pose_1 = Lister.list_to_pose(pose_list_1)
    pose_2 = Lister.list_to_pose(pose_list_2)

    joint_state_1 = Lister.list_to_joint_state(joint_state_list_1, "a")
    joint_state_2 = Lister.list_to_joint_state(joint_state_list_2, "pe")

    float32_multi_array_1 = Lister.list_to_given_data_field(
        float32_multi_array_list_1, Float32MultiArray
    )
    int32_1 = Lister.list_to_given_data_field(int32_list, Int32)

    print("Done")
