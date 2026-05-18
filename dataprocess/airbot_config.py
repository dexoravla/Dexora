from dataclasses import dataclass
from typing import Dict

@dataclass
class AirbotConfig:
    # 基本配置
    overwrite = True
    fps = 20.0  # 20Hz采样频率
    robot = "airbot_dexterous"
    video_backend = "pyav"
    
    # 数据路径配置
    source_data_root = "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/all"  # 需要设置实际路径
    output_data_root = "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/output"  # 需要设置实际路径
    log_root = "/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/all/logs"  # 需要设置实际路径

    # 数据集配置
    dataset_name = "airbot_dexterous_manipulation"  # 数据集名称
    task_name = "bimanual_dexterous_manipulation"   # 主任务名称
    
    # 任务配置 - 可根据不同数据集修改
    task_type = "pick_and_place_task"  # LeRobot的task参数 - 任务类型标识符
    default_instruction = "双臂灵巧手协同操作物体"  # 默认指令文本
    use_action_txt = True  # 是否使用action.txt文件中的指令，True则读取action.txt作为instruction
    
    # 处理配置
    num_image_writer_processes = 4
    num_image_writer_threads_per_camera = 4
    nonoop_threshold = 0.01
    
    
        # 设备标识配置
    device_id = "DSJ-2062-309"  # 从BSON中获取的设备ID
    device_type = "airbot"
    device_type_info = "具身双臂灵巧操作平台"
    station_id = "3784D4BA-87AF-47E7-B86D-42CA1904AA77"  # 从BSON中获取的站点ID
    device_version = "1.2.2"  # 从BSON中获取的版本信息
    driver_version = "1.0.0"
    
    # 相机具体型号配置
    camera_models = {
        "camera_high": "RGB Camera - Top View",
        "camera_left": "RGB Camera - Left Side", 
        "camera_right": "RGB Camera - Right Side",
        "camera_front": "RGB Camera - Front Head"  # DSJ-2062-309
    }
    
    # 机械臂具体型号配置
    arm_models = {
        "left_arm": "6DOF Manipulator - Left",
        "right_arm": "6DOF Manipulator - Right"
    }
    
    # 灵巧手具体型号配置
    hand_models = {
        "left_hand": "12DOF Dexterous Hand - Left", 
        "right_hand": "12DOF Dexterous Hand - Right"
    }
    
    # 相机标定信息配置
    camera_calibration = {
        "camera_high": {  # top相机
            "fx": None,  # 焦距x 
            "fy": None,  # 焦距y 
            "cx": None,  # 主点x 
            "cy": None,  # 主点y 
            "k1": None,  # 径向畸变k1 
            "k2": None,  # 径向畸变k2 
            "k3": None,  # 径向畸变k3 
            "p1": None,  # 切向畸变p1 
            "p2": None   # 切向畸变p2 
        },
        "camera_left": {
            "fx": None, "fy": None, "cx": None, "cy": None,
            "k1": None, "k2": None, "k3": None, "p1": None, "p2": None
        },
        "camera_right": {
            "fx": None, "fy": None, "cx": None, "cy": None,
            "k1": None, "k2": None, "k3": None, "p1": None, "p2": None
        },
        "camera_front": {
            "fx": None, "fy": None, "cx": None, "cy": None,
            "k1": None, "k2": None, "k3": None, "p1": None, "p2": None
        }
    }
    
    # 全局标定信息
    reprojection_error = None  # 全局重投影误差 - 需要实际标定数据
    calibration_date = None     # 标定日期 - 可选配置
    calibration_source = "calibration_tool"  # 标定工具来源
    # 设备数据收集配置
    data_collection_info = {
        "operator": "manual",
        "environment": "lab",
        "data_collection_method": "human_demonstration"
    }
    
    # 相机配置 (4个相机)
    rgb_dirs = [
        'camera_high',    # camera_2
        'camera_left',    # camera_0
        'camera_right',   # camera_6
        'camera_front'    # head_camera from BSON
    ]
    
    # 不同action的相机文件夹映射配置
    # 如果某个action没有配置，则使用default配置
    action_camera_mappings = {
        "default": {  # 默认映射
            'camera_high': 'camera_4',
            'camera_left': 'camera_2', 
            'camera_right': 'camera_6'
        },
        "action4": {  # action4的特定映射
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6'
        },
        "action27": {  # action27的特定映射
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'
        },
        "action8":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'                      
        },
        "action9":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'                      
        },
        "action10":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'                      
        },
        "action11":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'                      
        },
        "action13":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                   
        },
        "action15":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                     
        },
        "action17":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                        
        },
        "action18":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                     
        },
        "action19":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                      
        },
        "action20":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                     
        },
        "action21":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                     
        },
        "action22":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                     
        },
        "action23":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                     
        },
        "action24":{
            'camera_front': 'camera_0',
            'camera_high': 'camera_4',
            'camera_left': 'camera_2',
            'camera_right': 'camera_6',                     
        },
        "action27":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'                      
        },
        "action28":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'   
        },
        "action29":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'   
        },
        "action30":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'   
        },
        "action31":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'   
        },
        "action32":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'   
        },
        "action33":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'   
        },
        "action34":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'   
        },
        "action35":{
            'camera_high': 'camera_4',
            'camera_left': 'camera_11',
            'camera_right': 'camera_6',
            'camera_front': 'camera_0'   
        },
        "action37":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action38":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action39":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action40":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action41":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action42":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action43":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action44":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action45":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action46":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action47":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action48":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action49":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action50":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action51":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action52":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action53":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action54":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action55":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action56":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action57":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action58":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action59":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action60":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action61":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action62":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action63":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action64":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action65":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action66":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action67":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action68":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action69":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action70":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action71":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action72":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action73":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action74":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action75":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action76":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action77":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action78":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action79":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action80":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action81":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action82":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action83":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action84":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action85":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action86":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action87":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_6',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action98":{
            'camera_high': 'camera_8',
            'camera_left': 'camera_10',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action99":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action100":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action101":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action102":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action103":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action104":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action105":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action106":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action107":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'   
        },
        "action108":{
            "camera_high": "camera_2",
            "camera_left": "camera_0",
            "camera_right": "camera_4",
            "camera_front": "camera_8"
        },
        "action109":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'
        },
        "action110":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'
        },
        "action111":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'
        },
        "action112":{
            'camera_high': 'camera_10',
            'camera_left': 'camera_8',
            'camera_right': 'camera_12',
            'camera_front': 'camera_0'
        },
        "action113":{
            'camera_high': 'camera_16',
            'camera_left': 'camera_14',
            'camera_right': 'camera_18',
            'camera_front': 'camera_0'
        },
        "action114":{
            'camera_high': 'camera_16',
            'camera_left': 'camera_14',
            'camera_right': 'camera_18',
            'camera_front': 'camera_0'
        },
        "action115":{
            'camera_high': 'camera_16',
            'camera_left': 'camera_14',
            'camera_right': 'camera_18',
            'camera_front': 'camera_0'
        },
        "action116":{
            'camera_high': 'camera_16',
            'camera_left': 'camera_14',
            'camera_right': 'camera_18',
            'camera_front': 'camera_0'
        },
        "action117":{
            'camera_high': 'camera_16',
            'camera_left': 'camera_14',
            'camera_right': 'camera_18',
            'camera_front': 'camera_0'
        },
        "action118":{
            'camera_high': 'camera_16',
            'camera_left': 'camera_14',
            'camera_right': 'camera_18',
            'camera_front': 'camera_0'
        },
        "action122":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action122":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action123":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action124":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action125":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action126":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action127":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action128":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action129":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action130":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        
        "action131":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action132":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action133":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action134":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action135":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action136":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action137":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action138":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action139":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action140":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action141":{
            'camera_high': 'camera_13',
            'camera_left': 'camera_9',
            'camera_right': 'camera_15',
            'camera_front': 'camera_0'
        },
        "action142":{
            'camera_high': 'camera_high',
            'camera_left': 'camera_left',
            'camera_right': 'camera_right',
            'camera_front': 'camera_front'
        }
        # 可以继续添加其他action的映射
        # "action5": {
        #     'camera_high': 'camera_3',
        #     'camera_left': 'camera_1',
        #     'camera_right': 'camera_5'
        # }
    }
    
    rgb_names = [
        {'observation.images.camera_high': {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 20.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }},
        {'observation.images.camera_left': {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 20.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }},
        {'observation.images.camera_right': {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 20.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }},
        {'observation.images.camera_front': {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 20.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }}
    ]
    
    # 状态空间配置 (36维: 左臂6+左手12+右臂6+右手12)
    state_len = 36
    state_names = [
        # 左臂6个关节观测
        'left_arm_joint_1_obs', 
        'left_arm_joint_2_obs',
        'left_arm_joint_3_obs',
        'left_arm_joint_4_obs', 
        'left_arm_joint_5_obs', 
        'left_arm_joint_6_obs',
        # 左手12个关节观测
        'left_hand_joint_1_obs',
        'left_hand_joint_2_obs', 
        'left_hand_joint_3_obs',
        'left_hand_joint_4_obs', 
        'left_hand_joint_5_obs', 
        'left_hand_joint_6_obs',
        'left_hand_joint_7_obs', 
        'left_hand_joint_8_obs',
        'left_hand_joint_9_obs',
        'left_hand_joint_10_obs', 
        'left_hand_joint_11_obs', 
        'left_hand_joint_12_obs',
        # 右臂6个关节观测
        'right_arm_joint_1_obs', 
        'right_arm_joint_2_obs', 
        'right_arm_joint_3_obs',
        'right_arm_joint_4_obs', 
        'right_arm_joint_5_obs', 
        'right_arm_joint_6_obs',
        # 右手12个关节观测
        'right_hand_joint_1_obs', 
        'right_hand_joint_2_obs', 
        'right_hand_joint_3_obs',
        'right_hand_joint_4_obs', 
        'right_hand_joint_5_obs', 
        'right_hand_joint_6_obs',
        'right_hand_joint_7_obs', 
        'right_hand_joint_8_obs',
        'right_hand_joint_9_obs',
        'right_hand_joint_10_obs',
        'right_hand_joint_11_obs', 
        'right_hand_joint_12_obs'
    ]
    
    # 动作空间配置 (36维: 左臂6+左手12+右臂6+右手12)
    action_len = 36
    action_names = [
        # 左臂6个关节
        'left_arm_joint_1', 
        'left_arm_joint_2',
        'left_arm_joint_3',
        'left_arm_joint_4', 
        'left_arm_joint_5', 
        'left_arm_joint_6',
        # 左手12个关节
        'left_hand_joint_1',
        'left_hand_joint_2', 
        'left_hand_joint_3',
        'left_hand_joint_4', 
        'left_hand_joint_5', 
        'left_hand_joint_6',
        'left_hand_joint_7', 
        'left_hand_joint_8',
        'left_hand_joint_9',
        'left_hand_joint_10', 
        'left_hand_joint_11', 
        'left_hand_joint_12',
        # 右臂6个关节
        'right_arm_joint_1', 
        'right_arm_joint_2', 
        'right_arm_joint_3',
        'right_arm_joint_4', 
        'right_arm_joint_5', 
        'right_arm_joint_6',
        # 右手12个关节
        'right_hand_joint_1', 
        'right_hand_joint_2', 
        'right_hand_joint_3',
        'right_hand_joint_4', 
        'right_hand_joint_5', 
        'right_hand_joint_6',
        'right_hand_joint_7', 
        'right_hand_joint_8',
        'right_hand_joint_9',
        'right_hand_joint_10',
        'right_hand_joint_11', 
        'right_hand_joint_12'
    ]
    
    # BSON文件配置
    robot_bson_name = "episode_0.bson"
    hand_bson_name = "xhand_control_data.bson"
    
    # 设备信息
    device_info = {
        "camera_high": {
            "type": "RGB_Camera",
            "resolution": "640x480"
        },
        "camera_left": {
            "type": "RGB_Camera", 
            "resolution": "640x480"
        },
        "camera_right": {
            "type": "RGB_Camera",
            "resolution": "640x480"
        },
        "camera_front": {
            "type": "RGB_Camera",
            "resolution": "640x480"
        },
        "left_arm": {
            "type": "6DOF_ARM",
            "joints": 6,
            "dimension": {
                'left_arm_joint_1': 'rad',
                'left_arm_joint_2': 'rad',
                'left_arm_joint_3': 'rad',
                'left_arm_joint_4': 'rad',
                'left_arm_joint_5': 'rad',
                'left_arm_joint_6': 'rad'
            }
        },
        "right_arm": {
            "type": "6DOF_ARM",
            "joints": 6,
            "dimension": {
                'right_arm_joint_1': 'rad',
                'right_arm_joint_2': 'rad',
                'right_arm_joint_3': 'rad',
                'right_arm_joint_4': 'rad',
                'right_arm_joint_5': 'rad',
                'right_arm_joint_6': 'rad'
            }
        },
        "left_hand": {
            "type": "DEXTEROUS_HAND",
            "dof": 12,
            "has_force_feedback": True,
            "dimension": {
                "left_hand_joint_1": "rad", 
                "left_hand_joint_2": "rad",
                "left_hand_joint_3": "rad",
                "left_hand_joint_4": "rad",
                "left_hand_joint_5": "rad",
                "left_hand_joint_6": "rad",
                "left_hand_joint_7": "rad",
                "left_hand_joint_8": "rad",
                "left_hand_joint_9": "rad",
                "left_hand_joint_10": "rad",
                "left_hand_joint_11": "rad",
                "left_hand_joint_12": "rad"
            }
        },
        "right_hand": {
            "type": "DEXTEROUS_HAND",
            "dof": 12,
            "has_force_feedback": True,
            "dimension": {
                "right_hand_joint_1": "rad",  
                "right_hand_joint_2": "rad",
                "right_hand_joint_3": "rad",
                "right_hand_joint_4": "rad",
                "right_hand_joint_5": "rad",
                "right_hand_joint_6": "rad",
                "right_hand_joint_7": "rad",
                "right_hand_joint_8": "rad",
                "right_hand_joint_9": "rad",
                "right_hand_joint_10": "rad",
                "right_hand_joint_11": "rad",
                "right_hand_joint_12": "rad"
            }
        }
    }
    
    # 相机内参
    camera_intrinsic = {
        "camera_high": {
            "fx": 640.0,
            "fy": 480.0,
            "cx": 320.0,
            "cy": 240.0,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0
        },
        "camera_left": {
            "fx": 640.0,
            "fy": 480.0,
            "cx": 320.0,
            "cy": 240.0,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0
        },
        "camera_right": {
            "fx": 640.0,
            "fy": 480.0,
            "cx": 320.0,
            "cy": 240.0,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0
        },
        "camera_front": {
            "fx": 640.0,
            "fy": 480.0,
            "cx": 320.0,
            "cy": 240.0,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0
        }
    }
    
    # 设备episode映射
    device_episode = {
        "device_id": "airbot_play_001",
        "episodes": []
    }
    
    # 数据标注
    data_annotation = {
        "dataset_name": "airbot_dexterous_manipulation",
        "main_task": "bimanual_dexterous_manipulation",  # 与 task_name 保持一致
        "description": "双臂灵巧手协同操作数据集，包含多种操作任务类型",
        "success_rate": 1.0,
        "difficulty": "hard",
        "environment": "lab",
        "data_collection_method": "human_demonstration"
    }
    
    def get_camera_mapping(self, action_name: str) -> Dict[str, str]:
        """根据action名称获取相机映射"""
        if action_name in self.action_camera_mappings:
            return self.action_camera_mappings[action_name]
        else:
            return self.action_camera_mappings["default"]
    # Action-Instruction 精确映射
    action_instruction_mapping = {
        'action8':'First, pick up the medicine bottle with the left hand and place it on the white platform. Then, pick up the other white medicine bottle with the right hand and place it on the white platform.',
        'action9':'First, pick up the mouse box with the left hand and place it onto the round plate. Then, pick up the sponge with the left hand and place it onto the round plate.',
        'action10':'First, pick up the calculator box with the left hand and place it into the yellow bucket. Then, pick up the power bank box with the right hand and place it into the yellow bucket.',
        'action11':'First, pick up the rectangular block from the plate with the left hand. Then, pick up the rectangular block from the plate with the right hand.',
        'action13':'Put the potato in the left-side grid cell with the right hand.',
        'action15':'Put the pumpkin in the right-side grid cell with the right hand.',
        'action17':'First, pick up the lemon with the left hand and place it in the left-side grid cell. Then, pick up the orange with the right hand and place it in the right-side grid cell.',
        'action18':'First, pick up the pomegranate with the left hand and place it in the left-side grid cell. Then, pick up the mango with the right hand and place it in the right-side grid cell.',
        'action19':'Place the small bowl into the large bowl with the left hand.',
        'action20':'Place the pumpkin into the white large bowl with the left hand.',
        'action21':'Pick up the yellow pepper with the right hand and throw it into the blue bowl.',
        'action22':'First, pick up the pomegranate with the left hand and place it in the left-side grid cell. Then, pick up the mango with the right hand and place it in the right-side grid cell.',
        'action23':'First, pick up the potato with the left hand and place it in the left-side grid cell. Then, pick up the pumpkin with the right hand and place it in the right-side grid cell.',           
        'action24':'First, pick up the peach with the left hand and place it into the left grid cell. Then, pick up the pear with the right hand and place it into the right grid cell.',
        'action27':'First, pick up the large bowl with the left hand and the small bowl with the right hand. Then, place both bowls into the yellow bucket.',
        'action28':'First, pick up the umbrella with the left hand and place it on the white lid. Then, pick up the blue bottle with the right hand and place it on the white lid.',
        'action29':'First, pick up the tissue box with the left hand and place it on the white lid. Then, pick up the milk carton with the straw with the right hand and place it on the white lid.',
        'action30':'First, pick up the mouse box with the left hand and place it into the box. Then, pick up the blue network cable with the right hand and place it into the box.',
        'action31':'Then, place both the snacks and the tape measure onto the platform.',
        'action32':'First, pick up the red toy car with the left hand and place it on the round lid. Then, pick up the blue-packaged biscuits with the right hand and place them on the round lid.',
        'action33':'First, pick up the remote-controlled gripper with the left hand and the water bottle with the right hand. Then, place the remote-controlled gripper into the box with the left hand, followed by placing the water bottle into the box with the right hand.',
        'action34':'First, pick up the wet wipes with the left hand and place them onto the white platform. Then, pick up the arched building blocks with the right hand and place them onto the white platform.',
        'action35':'First, pick up the Rubik\'s cube with the right hand and place it onto the round plate. Then, pick up the paper cup with the left hand and place it on top of the',
        'action37':'First, pick up the calculator case with the left hand and the power bank case with the right hand. Then, place the calculator case with the left hand, followed by placing the power bank case with the right hand.',
        'action38':'First, pick up the identical white cups with both hands. Then, place the cup with the left hand into the yellow bucket, followed by placing the cup with the right hand into the yellow bucket.',
        'action39':'First, pick up the blue large bowl with the right hand and place it on the square plate. Then, pick up the wet wipes with the left hand and place them into the blue bowl.',
        'action40':'First, pick up the rectangular block with the left hand and throw it into the square plate. Then, pick up the rectangular block with the right hand and throw it into the square plate.',
        'action41':'First, pick up the white cup from the white lid with the left hand. Then, pick up the other white cup from the white lid with the right hand.',
        'action42':'First, pick up the Vitamin B bottle from the white lid with the left hand. Then, pick up the other Vitamin B bottle from the white lid with the right hand.',
        'action43':'First, pick up the white umbrella from the white lid with the right hand. Then, pick up the bottle of mineral water from the white lid with the left hand.',
        'action44':'Pick up the water bottle from the left side of the table with the left hand and pass it to the right hand. The right hand then places the water bottle on the right side of the table.',
        'action45':'Pick up the calculator box from the left side of the table with the left hand and pass it to the right hand. The right hand then places the calculator box on the right side of the table.',
        'action46':'Pick up the white lid from the left side of the table with the left hand and pass it to the right hand. The right hand then places the white lid on the right side of the table.',
        'action47':'First, use the left hand to take the sponge out of the white lid and place it on the table. Then, use the right hand to take the bowl out of the white lid and place it on the table.',
        'action48':'First, use the left hand to take the drink out of the white lid and place it on the table. Then, use the right hand to take the coffee cup out of the white lid and place it on the table.',
        'action49':'First, use the left hand to take the building block out of the plate and place it on the table. Then, use the right hand to take the BB pellet out of the plate and place it on the table.',
        'action50':'First, use the left hand to throw the BB pellet into the bowl. Then, use the right hand to throw the building block into the bowl.',
        'action51':'First, use the left hand to take the green toy car out of the plate and place it on the table. Then, use the right hand to take the yellow toy car out of the plate and place it on the table.',
        'action52':'First, use the left hand to take one pack of tissues out of the white tray and place it on the table. Then, use the right hand to take the other pack of tissues out of the white tray and place it on the table.',
        'action53':'First, use the left hand to put one spoon into the basin. Then, use the right hand to put the other spoon into the basin.',
        'action54':'First, use the left hand to pick up a pack of tissues from the table and put it into the basin. Then, use the right hand to pick up another pack of tissues from the table and put it into the basin.',
        'action55':'First, use the left hand to take the sponge out of the plate and place it on the table. Then, use the right hand to pick up the mouse box from the table and place it into the plate.',
        'action56':'First, use the left hand to take the drink out of the plate and place it on the table. Then, use the right hand to pick up the wet wipes from the table and place them into the plate.',
        'action57':'First, use the left hand to take the BB pellet out of the plate and place it on the table. Then, use the right hand to pick up the building block from the table and place it into the plate.',
        'action58':'Use both hands to pick up the steel frame and place it on the table.',
        'action59':'First, use the left hand to pick up the hand piano and place it on the white lid. Then, use the right hand to pick up the needle-nose pliers and place them on the white lid.',
        'action60':'First, use the left hand to pick up the pliers and place them into the white lid. Then, use the right hand to pick up the utility knife and place it into the white lid.',
        'action61':'First, use the left hand to pick up the small umbrella and place it into the white lid. Then, use the right hand to pick up the ruler and place it into the white lid.',
        'action62':'Use the right hand to pick up the cup and pour the BB pellets inside into the bowl.',
        'action63':'First, use the left hand to pick up the water bottle from the white lid and place it on the table. Then, use the right hand to pick up the tape measure from the white lid and place it on the table.',
        'action64':'Pick up one coffee cup with the left hand and put it into the basin. Pick up the other coffee cup with the right hand and put it into the basin.',
        'action65':'Pick up the calculator box from the table with the left hand and pass it to the right hand. The right hand receives the calculator box and puts it into the white lid.',
        'action66':'Pick up the book from the table with the left hand and pass it to the right hand. The right hand receives the book and puts it on top of another book.',
        'action67':'Pick up the building block from the table with the left hand and put it into the bowl. Pick up the tape measure from the table with the right hand and put it into the small plate.',
        'action68':'Simultaneously use the left and right hands to pick up the umbrella and tissues from the table and put them into the white lid.',
        'action69':'Hold the laptop with the left hand and close it with the right hand.',
        'action70':'Simultaneously use the left and right hands to pick up the sponge and wet wipes from the table and put them into the white plate.',
        'action71':'Simultaneously use the left and right hands to pick up the mouse box and power bank box from the table and put them into the white lid.',
        'action72':'Simultaneously use the left and right hands to pick up the cakes from the table and put them into the white plate.',
        'action73':'First, use the left hand to pick up the cake from the plate and place it on the table. Then, use the right hand to pick up the breadstick from the table and place it into the plate.',
        'action74':'Simultaneously use the left and right hands to pick up the building blocks from the table and put them into the plate.',
        'action75':'First, use the left hand to stack the mouse box on top of the calculator box. Then, use the right hand to put the coffee cup into the plate.',
        'action76':'Use both hands to simultaneously take two black cakes out of the white box and place them on the table.',
        'action77':'First, use the left hand to pick up a Swiss roll and put it into the plate. Then, use the right hand to pick up the other Swiss roll and put it into the plate.',
        'action78':'Use both hands to simultaneously pick up the shark plush toys from the table and put them into the white lid.',
        'action79':'First, use the left hand to pick up an ice cream and put it into the small white basin. Then, use the right hand to pick up the other ice cream and put it into the small white basin.',
        'action80':'Use both hands to simultaneously pick up the dog plush toys from the white plate and then place them on the table.',
        'action81':'Pick up the shark plush toy from the table with the left hand and pass it to the right hand. The right hand receives the shark plush toy and puts it into the white lid.',
        'action82':'Simultaneously use the left and right hands to pick up the cake and ice cream from the table, then simultaneously throw the cake into the bowl and the ice cream into the plate.',
        'action83':'First, use the left hand to take the sponge out of the plate and place it on the table. Then, use the right hand to pick up the cake from the table and place it into the plate.',
        'action84':'First, use the left hand to move the large building block from the left side of the table to the center. Then, use the right hand to stack the small building block from the right side of the table on top of the large building block.',
        'action85':'Pick up the syringe from the table with the left hand, and push the plunger with the right hand.',
        'action86':'First, use the left hand to pick up a building block from the table and place it on top of another building block. Then, use the right hand to pick up another building block from the table and place it on top of the building blocks.',
        'action87':'First, use the left hand to pick up a 6cm cube and put it into the pink plate on the left. Then, use the right hand to put the other 6cm cube into the white plate on the right.',
        'action97':'Simultaneously use both hands to pick up the blue and green cubes from inside the paper box lid and place them on the table.',
        'action98':'First, use the left hand to take the BB pellet out of the plate and place it on the table. Then, use the right hand to pick up the cake from the table and place it into the plate.',
        'action99':'Pick up the syringe from the table with the left hand, pull the plunger with the right hand, then use the left hand to put the syringe back on the table.',
        'action100':'First, use the left hand to place the yellow block from the table onto the Rubik\'s cube on the table. Then, use the right hand to take the yellow block from the Rubik\'s cube and place it back on the table.',
        'action101':'Simultaneously use the left and right hands to place the small cubes from the left and right sides onto the corresponding large cubes on the left and right sides.',
        'action102':'First, use the left hand to pick up the biscuit from the table and put it into the white lid. Then, use the right hand to put the beer mug from the table into the white lid.',
        'action103':'Simultaneously use both hands to pick up the shark daggers from the table and place them on the white lid.',
        'action104':'Pick up the large phone from the table with the left hand and place it on the calculator box. Then, pick up the large phone from the calculator box with the right hand and place it on the table.',
        'action105':'Pick up the cake from the table with the left hand and place it into the plate. Then, pick up the cake from the plate with the right hand and place it on the table.',
        'action106':'First, use the left hand to pick up the purple cube from the table and put it into the plate. Then, use the right hand to pick up the blue cube from the table and place it on top of the purple cube.',
        'action107':'First, use the left hand to pick up the cake from the table and put it into the white lid. Then, use the right hand to pick up the ice cream from the table and put it into the white lid. Finally, use the left hand to pick up the other cake from the table and put it into the white lid.',
        'action108':'Pick up the cake from the table with the left hand and place it on the yellow cube. Then, pick up the other cake from the table with the right hand and place it on the blue cube.',
        'action109':'Pick up the egg from the table with the left hand and place it into the plate. Then, pick up the other egg from the table with the right hand and place it into the plate.',
        'action110':'Pick up the shark plush toy from the table with the left hand and place it into the paper box lid. Then, pick up the gold bar from the table with the right hand and place it into the paper box lid.',
        'action111':'Pick up the book from the table with the right hand and place it on the bookshelf.',
        'action112':'Pick up the braised pork from the table with the left hand and place it into the plate. Then, pick up the prawn from the table with the right hand and place it into the plate.',
        'action113':'Push the book to the left side of the table with the right hand, then pick up the book with the left hand and place it on the bookshelf.',
        'action114':'Push the frying pan to the right side of the table with the left hand, then pick up the frying pan with the right hand and place it on the red cube.',
        'action115':'First, use the left hand to pick up the double-sided tape from the table and put it on the Mirinda bottle. Then, use the right hand to pick up another double-sided tape from the table and put it on the Mirinda bottle.',
        'action116':'Pick up the blue cup from the left side of the table with the left hand and put it into the plate. Then, pick up the purple cup from the right side of the table with the right hand and put it into the blue cup.',
        'action117':'Pick up the diamond from the table with the right hand and put it into the box. Then, use the left hand to close the lid of the box.',
        'action118':'Pick up the egg from the table with the right hand and put it into the egg box.',
        'action122':'Pick up the yellow egg with the right hand.',
        'action123':'Pick up the knife from the knife rack with the right hand, hold the green onion on the table with the left hand, cut the green onion with the right hand, then put the knife back on the knife rack.',
        'action124':'Pick up the small guitar from the table with the left hand, pluck the strings with the right hand, then put down the small guitar with the left hand.',
        'action125':'Pick up the mini bowl can from the table with the right hand and place it on the mini table.',
        'action126':'Pick up the egg from the table with the right hand and put it into the egg box. Then, use the left hand to close the lid of the egg box.',
        'action127':'Pick up the cake on the left with the left hand and place it on the wooden rack. Pick up the cake on the right with the right hand and place it on the wooden rack.',
        'action128':'Pick up the glasses case from the left side of the table with the left hand and place it in the center of the table. Then, pick up the gold bar from the right side of the table with the right hand and place it on the glasses case.',
        'action129':'Pick up the three cups from the left side of the table with the right hand and place them on the book on the left side of the table.',
        'action130':'Pick up the egg on the left with the left hand and place it into the bowl in the center. Pick up the egg on the right with the right hand and place it into the bowl in the center.',
        'action131':'Pick up the water bottle on the table with the left hand, unscrew the cap with the right hand, then place the water bottle back on the table with the left hand.',
        'action132':'Pick up the racket with the left hand, pick up the ball with the right hand and place it on the racket, then place the racket on the table with the left hand.',
        'action133':'Pick up the yellow square block on the left side of the table with the left hand and put it into the blue toy ring. Then, pick up the yellow square block on the right side of the table with the right hand and put it into the blue toy ring.',
        'action134':'Use the right index finger to hook <CENTURY> down from the bookshelf.',
        'action135':'Pick up the green cylinder on the left side of the table with the left hand and put it into the paper box. Then, pick up the red cylinder on the right side of the table with the right hand and put it into the paper box.',
        'action136':'Pick up the blue square block on the left side of the table with the left hand and place it on top of the yellow square block in the center of the table. Then, pick up the green square block on the right side of the table with the right hand and place it on top of the blue square block.',
        'action137':'Use both hands to simultaneously grab the basin on the table, lift it up, and put it down.',
        'action138':'Push the arched building block to the upper right position with the right hand.',
        'action139':'Push the dark blue and yellow square blocks to the designated position under the arched building block at the upper right with the right hand.',
        'action138':'Push the arched building block to the upper right position with the right hand.',
        'action139':'Push the dark blue and yellow square blocks to the designated position under the arched building block at the upper right with the right hand.',
        'action140':'Use the right hand to push the red and green cylindrical building blocks to the designated position under the right square building block.',
        'action141':'First, pick up the blue square block from the table with the left hand and place it on the small table. Then, pick up the red square block from the table with the right hand and place it on top of the blue square block.',
        'action142':'First, pick up the red rectangular block from the left side of the table with the left hand and place it in the center of the table. Then, pick up the blue hexagonal block from the right side of the table with the right hand and place it on top of the red rectangular block.',
    }
    
    # 任务分类映射
    task_categories = {
        'fruit_sorting': {  
            'actions': ['action8','action9','action10','action11','action13','action15','action17','action18','action19','action20','action21','action22','action23','action24','action25','action27',
                        'action28','action29','action30','action31','action32','action33','action34','action35','action37','action38','action39','action40','action41','action42','action43',
                        'action44','action45','action46','action47','action48','action49','action50','action51','action52','action53','action54','action55','action56','action57','action58',
                        'action59','action60','action61','action62','action63','action64','action65','action66','action67','action68','action69','action70','action71','action72','action73',
                        'action74','action75','action76','action77','action78','action79','action80','action81','action82','action83','action84','action85','action86','action87','action97',
                        'action98','action99','action100','action101','action102','action103','action104','action105','action106','action107','action108','action109','action110','action111',
                        'action112','action113','action114','action115','action116','action117','action118','action122','action123','action124','action125','action126','action127','action128',
                        'action129','action130','action131','action132','action133','action134','action135','action136','action137','action138','action139','action140'
                        ],
            'task_index': 0,
            'task_name': 'pick_place_task',  # 子任务名称
            'description': 'pick and place'  # 子任务描述
        },
 
    }
    