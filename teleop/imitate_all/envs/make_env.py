from typing import List
from robots.common_robot import AssembledRobot


def make_environment(env_config):
    robot_name = env_config["robot_name"]
    habitat = env_config["habitat"]
    robot_num = env_config["robot_num"]
    joint_num = env_config["joint_num"]
    start_joint = env_config["start_joint"]
    fps = env_config["fps"]
    can_buses = env_config["can_buses"]
    eef_mode = env_config["eef_mode"]
    bigarm_type = env_config["bigarm_type"]
    forearm_type = env_config["forearm_type"]
    check_images = env_config["check_images"]

    robot_name = robot_name if not check_images else "fake_robot"

    assert (
        len(start_joint) == robot_num * joint_num
    ), "The length of start_joint should be equal to joint_num or joint_num*robot_num"
    print(f"Start joint: {start_joint}")

    robot_instances: List[AssembledRobot] = []
    env = None
    if robot_name == "none" or habitat == "mujoco":
        print("No direct robot is used")
    elif "airbot_play" in robot_name:
        # set up can
        # from utils import CAN_Tools
        import airbot

        vel = 2.0
        for i in range(robot_num):
            # if 'v' not in can:
            #     if not CAN_Tools.check_can_status(can):
            #        success, error = CAN_Tools.activate_can_interface(can, 1000000)
            #        if not success: raise Exception(error)
            airbot_player = airbot.create_agent(
                "/usr/share/airbot_models/airbot_play_with_gripper.urdf",
                "down",
                can_buses[i],
                vel,
                eef_mode[i],
                bigarm_type[i],
                forearm_type[i],
            )
            robot_instances.append(
                AssembledRobot(
                    airbot_player,
                    1 / fps,
                    start_joint[joint_num * i : joint_num * (i + 1)],
                )
            )
    elif "fake" in robot_name or "none" in robot_name:
        from robots.common_robot import AssembledFakeRobot

        if check_images:
            AssembledFakeRobot.real_camera = True
        for i in range(robot_num):
            robot_instances.append(
                AssembledFakeRobot(
                    1 / fps, start_joint[joint_num * i : joint_num * (i + 1)]
                )
            )
    elif "ros" in robot_name:
        from robots.common_robot import AssembledRosRobot
        import rospy

        rospy.init_node("replay_episodes")
        namespace = "/airbot_play"
        states_topic = f"{namespace}/joint_states"
        arm_action_topic = f"{namespace}/arm_group_position_controller/command"
        gripper_action_topic = f"{namespace}/gripper_group_position_controller/command"
        for i in range(robot_num):
            robot_instances.append(
                AssembledRosRobot(
                    states_topic,
                    arm_action_topic,
                    gripper_action_topic,
                    joint_num,
                    start_joint[joint_num * i : joint_num * (i + 1)],
                    1 / fps,
                )
            )
    elif "mmk" in robot_name:
        from robots.common_robot import AssembledMmkRobot

        for i in range(robot_num):
            robot_instances.append(AssembledMmkRobot())
    else:
        raise NotImplementedError(f"{robot_name} is not implemented")

    if habitat == "real":
        if "airbot_play" in robot_name:
            from envs.airbot_play_real_env import make_env
        elif "fake" in robot_name:
            # from airbot_play_fake_env import make_env  # TODO: implement this or pass some param to make_env
            from envs.airbot_play_real_env import make_env
        elif "ros" in robot_name:
            from envs.airbot_play_real_env import make_env
        elif "mmk" in robot_name:
            from envs.airbot_mmk_env import make_env
        else:
            raise NotImplementedError(f"robot_name: {robot_name} is not implemented")
    elif habitat == "mujoco":
        from envs.airbot_play_mujoco_env import make_env
        print(f"Using Mujoco environment with config file: {robot_name}")
        env = make_env(robot_name)
    elif habitat == "isaac":
        raise NotImplementedError(f"habitat: {habitat} is not implemented")
    else:
        raise NotImplementedError(f"habitat: {habitat} is not implemented")
    camera_names = env_config["camera_names"]
    camera_indices = env_config["camera_indices"]
    if camera_indices != "":
        cameras = {
            name: int(index) for name, index in zip(camera_names, camera_indices)
        }
    else:
        cameras = camera_names
    if env is None:
        env = make_env(robot_instance=robot_instances, cameras=cameras)
    env.set_reset_position(start_joint)
    return env
