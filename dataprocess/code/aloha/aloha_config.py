from dataclasses import dataclass


@dataclass
class DataConfig:
    overwrite = True

    # source_data_json = [
    #     '/home/liuyou/Documents/annotation/realman/realman构建块project-352-at-2025-07-08-05-49-bf414113.json'
    # ]
    # source_data_roots = [
    #     '/home/liuyou/Documents/dataset/realman/build_blocks/episode_1.hdf5',
    #     '/home/liuyou/Documents/dataset/realman/build_blocks/episode_2.hdf5',
    # ]
    source_data_csv = '/home/ctos/aloha/nas-trans.csv'
   
    rgb_dirs = [
        'cam_high',
        'cam_left_wrist',
        'cam_right_wrist'
    ]
    rgb_names = [
        {'observation.images.cam_high':{
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }},
        {'observation.images.cam_left_wrist':{
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }},
        {'observation.images.cam_right_wrist':{
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }},
    ]

    action_len = 28
   
    action_name = [
        'right_arm_joint_1',
        'right_arm_joint_2',
        'right_arm_joint_3',
        'right_arm_joint_4',
        'right_arm_joint_5',
        'right_arm_joint_6',
        'right_gripper_joint',
        'right_end_effector_positions_x',
        'right_end_effector_positions_y',
        'right_end_effector_positions_z',
        'right_end_effector_quat_x',
        'right_end_effector_quat_y',
        'right_end_effector_quat_z',
        'right_end_effector_quat_w',
        'left_arm_joint_1',
        'left_arm_joint_2',
        'left_arm_joint_3',
        'left_arm_joint_4',
        'left_arm_joint_5',
        'left_arm_joint_6',
        'left_gripper_joint',
        'left_end_effector_positions_x',
        'left_end_effector_positions_y',
        'left_end_effector_positions_z',
        'left_end_effector_quat_x',
        'left_end_effector_quat_y',
        'left_end_effector_quat_z',
        'left_end_effector_quat_w',
    ]
    action_index = [0,1,2,3,4,5,10,30,31,32,
                    33,34,35,36,37,38,50,51,
                    52,53,54,55,60,80,81,82,
                    83,84,85,86,87,88]
    nonoop_threshold = 1e-3


    data_root = '/home/ctos/lerobot/aloha/'
    hdf5_root = '/home/ctos/aloha/aloha/'
    fps=25
    video_backend = 'pyav'
    robot = 'aloha'
    num_image_writer_processes=0
    num_image_writer_threads_per_camera=4
    log_root='/home/ctos/aloha/aloha/log'
    device_info = {
        "device_list": [
        {
            "device_id": "8xK992pQ",
            "device_type" : "aloha",
            "device_type_info":"分体式ALOHA",
            "device_info": {
                "cam_high": {
                    "type": None
                },
                "cam_left_wrist":{
                    "type":None
                },
                "cam_right_wrist":{
                    "type":None
                },
                "piper_left":{
                    "type":"Piper-6DOF",
                    "dimension":{
                            'left_arm_joint_1':'rad',
                            'left_arm_joint_2':'rad',
                            'left_arm_joint_3':'rad',
                            'left_arm_joint_4':'rad',
                            'left_arm_joint_5':'rad',
                            'left_arm_joint_6':'rad',
                            'left_gripper_joint':'rad',
                            'left_end_effector_positions_x':'m',
                            'left_end_effector_positions_y':'m',
                            'left_end_effector_positions_z':'m',
                            'left_end_effector_quat_x':None,
                            'left_end_effector_quat_y':None,
                            'left_end_effector_quat_z':None,
                            'left_end_effector_quat_w':None,
                    },
                    "description":{
                            'left_arm_joint_1':None,
                            'left_arm_joint_2':None,
                            'left_arm_joint_3':None,
                            'left_arm_joint_4':None,
                            'left_arm_joint_5':None,
                            'left_arm_joint_6':None,
                            'left_gripper_joint':None,
                            'left_end_effector_positions_x':None,
                            'left_end_effector_positions_y':None,
                            'left_end_effector_positions_z':None,
                            'left_end_effector_quat_x':None,
                            'left_end_effector_quat_y':None,
                            'left_end_effector_quat_z':None,
                            'left_end_effector_quat_w':None,
                    }
                },
                "piper_right":{
                    "type":"Piper-6DOF",
                    "dimension":{
                            'right_arm_joint_1':'rad',
                            'right_arm_joint_2':'rad',
                            'right_arm_joint_3':'rad',
                            'right_arm_joint_4':'rad',
                            'right_arm_joint_5':'rad',
                            'right_arm_joint_6':'rad',
                            'right_gripper_joint':'rad',
                            'right_end_effector_positions_x':'m',
                            'right_end_effector_positions_y':'m',
                            'right_end_effector_positions_z':'m',
                            'right_end_effector_quat_x':None,
                            'right_end_effector_quat_y':None,
                            'right_end_effector_quat_z':None,
                            'right_end_effector_quat_w':None,
                    },
                    "description":{
                            'right_arm_joint_1':None,
                            'right_arm_joint_2':None,
                            'right_arm_joint_3':None,
                            'right_arm_joint_4':None,
                            'right_arm_joint_5':None,
                            'right_arm_joint_6':None,
                            'right_gripper_joint':None,
                            'right_end_effector_positions_x':None,
                            'right_end_effector_positions_y':None,
                            'right_end_effector_positions_z':None,
                            'right_end_effector_quat_x':None,
                            'right_end_effector_quat_y':None,
                            'right_end_effector_quat_z':None,
                            'right_end_effector_quat_w':None,
                    }
                }

            }
        }
    ]
    }

    # camera_info = {
    # "device_list": [
    #     {
    #         "device_id": "8xK992pQ",
    #         "calibration_info": {
    #             "version": "1.0",
    #             "date": "2025-07-17",
    #             "source": "calibration_tool",
    #             "reprojection_error": 0.8
    #         },
    #         "camera_intrinsics": {
    #             "top": {
    #                 "model": "PINHOLE",
    #                 "parameters": {
    #                     "width": 640,
    #                     "height": 480,
    #                     "fx": 605.849,
    #                     "fy": 605.742,
    #                     "cx": 309.673,
    #                     "cy":  245.429
    #                 },
    #                 "distortion": {
    #                     "model": "Inverse Brown Conrady",
    #                     "k1": 0,
    #                     "k2": 0,
    #                     "k3": 0,
    #                     "p1": 0,
    #                     "p2": 0
    #                 }
    #             },
    #             "right": {
    #                 "model": "PINHOLE",
    #                 "parameters": {
    #                     "width": 640,
    #                     "height": 480,
    #                     "fx": 606.736,
    #                     "fy": 605.451,
    #                     "cx": 322.341,
    #                     "cy": 255.021
    #                 },
    #                 "distortion": {
    #                     "model": "Inverse Brown Conrady",
    #                     "k1": 0,
    #                     "k2": 0,
    #                     "k3": 0,
    #                     "p1": 0,
    #                     "p2": 0
    #                 }
    #             },
    #             "left": {
    #                 "model": "PINHOLE",
    #                 "parameters": {
    #                     "width": 640,
    #                     "height": 480,
    #                     "fx": 607.072,
    #                     "fy": 607.943,
    #                     "cx": 315.199,
    #                     "cy": 245.262
    #                 },
    #                 "distortion": {
    #                     "model": "Inverse Brown Conrady",
    #                     "k1": 0,
    #                     "k2": 0,
    #                     "k3": 0,
    #                     "p1": 0,
    #                     "p2": 0
    #                 }
    #             }
    #         }
    #     }
    # ]
    # }


