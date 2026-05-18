import argparse
from dataclasses import dataclass, replace, field
from typing import Optional, Dict, List
import time
import logging
import numpy as np
from bson import BSON
import sys
from typing import Tuple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from airbot_py.airbot_mmk2 import AirbotMMK2
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from mmk2_types.types import (
    MMK2Components,
    JointNames,
    ComponentTypes,
    TopicNames,
    MMK2ComponentsGroup,
    ImageTypes,
    ControllerTypes,
)
from mmk2_types.grpc_msgs import (
    JointState,
    TrajectoryParams,
    MoveServoParams,
    ForwardPositionParams,
    JointState,
)

def load_bson(bson_file: str) -> dict:
    with open(bson_file, "rb") as f:
        data = BSON.decode(f.read())
    print(f"Loaded BSON data from {bson_file}")
    return data

def load_parquet_actions(parquet_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load actions (Nx36) and timestamps (N,) from a lerobot-style parquet.

    Returns:
        actions: np.ndarray of shape (N, 36)
        timestamps: np.ndarray of shape (N,)
    """
    try:
        import pandas as pd
    except Exception as e:
        print("ERROR: pandas is required to read parquet. Please `pip install pandas pyarrow`.")
        raise

    df = pd.read_parquet(parquet_file)
    if "actions" not in df.columns:
        raise ValueError("Parquet missing 'actions' column")
    # Optional timestamp
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy(dtype=np.float64)
    else:
        # fallback to 0..N-1 at 1/freq, actual freq applied later
        ts = np.arange(len(df), dtype=np.float64)

    # actions column is a sequence (list) of length 36 per row
    # Normalize to np.ndarray (N, 36)
    actions = np.stack(df["actions"].apply(lambda x: np.asarray(x, dtype=np.float32)).to_list(), axis=0)
    if actions.shape[1] != 36:
        raise ValueError(f"Expected actions dim=36, got {actions.shape[1]}")
    return actions, ts

@dataclass
class AIRBOTMMK2Config(object):
    name: str = "mmk2"
    domain_id: int = -1
    ip: str = "172.25.11.188"
    port: int = 50055
    default_action: Optional[List[float]] = field(
        default_factory=lambda: [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0,
            0.15
        ]
    )
    cameras: Dict[str, str] = field(default_factory=lambda: {})
    components: List[str] = field(
        default_factory=lambda: [
            MMK2Components.LEFT_ARM.value,
            # MMK2Components.LEFT_ARM_EEF.value,
            MMK2Components.RIGHT_ARM.value,
            # MMK2Components.RIGHT_ARM_EEF.value,
            MMK2Components.HEAD.value,
            MMK2Components.SPINE.value,
        ]
    )
    demonstrate: bool = False

class MMK2REPLAY(object):
    def __init__(self, config: Optional[AIRBOTMMK2Config] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTMMK2Config()
        self.config = replace(config, **kwargs)
        self.robot = AirbotMMK2(
            self.config.ip,
            self.config.port,
            self.config.name,
            self.config.domain_id,
        )
        self.joint_names = {}
        self.cameras: Dict[MMK2Components, str] = {}
        self.components: Dict[MMK2Components, ComponentTypes] = {}
        all_joint_names = JointNames()
        self.joint_num = 0
        for k, v in self.config.cameras.items():
            self.cameras[MMK2Components(k)] = ImageTypes(v)
        for comp_str in self.config.components:
            comp = MMK2Components(comp_str)
            # TODO: get the type info from SDK
            self.components[comp] = ComponentTypes.UNKNOWN
            names = all_joint_names.__dict__[comp_str]
            self.joint_names[comp] = names
            self.joint_num += len(names)
        print(f"Joint names: {self.joint_names}")    
        logger.info(f"Components: {self.components}")
        logger.info(f"Joint numbers: {self.joint_num}")
        self.robot.enable_resources(
            {
                comp: {
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "false",
                }
                for comp in self.cameras
            }
        )

        self.logs = {}
        self.enter_active_mode = lambda: self._set_mode("active")
        self.enter_passive_mode = lambda: self._set_mode("passive")
        self.get_state_mode = lambda: self._state_mode
        self.exit = lambda: None
        self.reset()
    
    def reset(self, sleep_time=0):
        if self.config.default_action is not None:
            goal = self._action_to_goal(self.config.default_action)
            self.robot.set_goal(goal, TrajectoryParams())
        else:
            logger.warning("No default action is set.")
        time.sleep(sleep_time)
        self.enter_servo_mode()

    def send_action(self, action, wait=False):
        goal = self._action_to_goal(action)
        if self.traj_mode:
            self.robot.set_goal(goal, TrajectoryParams())
        else:
            self.robot.set_goal(goal, MoveServoParams())

    def _set_mode(self, mode):
        self._state_mode = mode

    def _action_check(self, action):
        assert (
            len(action) == self.joint_num
        ), f"Invalid action {action} with length: {len(action)}"

    def _action_to_goal(self, action) -> Dict[MMK2Components, JointState]:
        self._action_check(action)
        goal = {}
        j_cnt = 0
        for comp in self.components:
            end = j_cnt + len(self.joint_names[comp])
            goal[comp] = JointState(position=action[j_cnt:end])
            j_cnt = end
        return goal

    def enter_traj_mode(self):
        self.traj_mode = True

    def enter_servo_mode(self):
        self.traj_mode = False

# 自动解析动作数据（BSON 路径）
def parse_actions_from_data(data, components, joint_names):
    """自动从BSON数据中解析动作序列"""
    all_actions = []
    
    # 获取数据长度（以第一个组件为准）
    first_component = list(components.keys())[0]
    # component_topic = f"/mmk/mmk/{first_component.value}/joint_state"
    component_topic = f"/observation/{first_component.value}/joint_state"
    data_length = len(data["data"][component_topic])
    
    print(f"数据长度: {data_length}")
    
    for i in range(data_length):
        action = []
        # 按照components的顺序自动提取各组件的位置数据
        for component in components:
            # component_topic = f"/mmk/mmk/{component.value}/joint_state"
            component_topic = f"/observation/{component.value}/joint_state"
            if component_topic in data["data"]:
                pos_data = data["data"][component_topic][i]["data"]["pos"]
                action.extend(pos_data)
            else:
                logger.warning(f"未找到组件 {component.value} 的数据")
        
        all_actions.append(action)
    
    return all_actions


def main():
    parser = argparse.ArgumentParser(description="MMK2 动作回放工具")
    parser.add_argument("file_path", nargs='?', default=None, help="BSON 数据文件路径（可选，若提供 --parquet 则忽略）")
    parser.add_argument("--parquet", default=None, help="从 lerobot Parquet 读取 actions 与 timestamp")
    parser.add_argument("--ip", default="192.168.11.200", help="机器人IP地址 (默认: 172.25.11.188)")
    parser.add_argument("--freq", type=float, default=20.0, help="回放频率 Hz (默认: 20.0)")
    parser.add_argument("--only-arms", action="store_true", default=True, help="仅启用左右臂组件 (默认: 开启)")
    args = parser.parse_args()
    # 确定数据源
    actions_arms = None
    timestamps = None

    # 仅启用左右臂组件，满足“头和脊柱静止”的需求
    components_override = None
    if args.only_arms:
        components_override = [
            MMK2Components.LEFT_ARM.value,
            MMK2Components.RIGHT_ARM.value,
        ]

    if args.parquet:
        actions_all, timestamps = load_parquet_actions(args.parquet)
        # 取前 12 维（左臂6+右臂6）
        actions_arms = actions_all[:, :12].astype(np.float32)
        mmk2 = MMK2REPLAY(ip=args.ip, components=components_override, default_action=[0.0]*12)
        mmk2.enter_servo_mode()
        all_actions = [a.tolist() for a in actions_arms]
    else:
        if not args.file_path:
            raise SystemExit("必须提供 BSON 文件路径或 --parquet 路径之一")
        file_path = args.file_path
        data = load_bson(file_path)
        mmk2 = MMK2REPLAY(ip=args.ip, components=components_override)
        mmk2.enter_servo_mode()
        all_actions = parse_actions_from_data(data, mmk2.components, mmk2.joint_names)

    freq = args.freq  # 使用外部传入的频率参数
    cnt = 100
    while cnt>=0 and len(all_actions) > 0:
        mmk2.send_action(all_actions[0])
        cnt-=1

    print("初始化完成，等待主控信号...")
    print("READY", flush=True)

    signal = sys.stdin.readline().strip()
    if signal.upper() == "START":
        print("收到 START，开始执行任务")
        for action in all_actions:
            start = time.perf_counter()
            mmk2.send_action(action)
            time.sleep(max(0, 1 / freq - (time.perf_counter() - start)))

if __name__ == "__main__":
    main()