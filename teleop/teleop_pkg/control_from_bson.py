
# 用户的使用流程： 
#   1. import XHandTeleOps
#   2. 实例化 XHandTeleOps
#   3. 调用它的方法：目前只提供三个方法
#           1. 从 versionpro 或 opentelevison 读数据
#           2. 调用 retarget_data 转化数据为 Xhand 灵巧手格式
#           3. 调用 send_data_xhand() 发送控制灵巧手移动
#           4. 调用 get_hand_full_info("hand_a") 或者 get_hand_full_info("hand_b") 读取手的位置，力矩等信息

# """以下是 receiver_main 的 demo 测试代码"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
from xhand_tele_ops import XHandTeleOps
from bson import BSON
import numpy as np
import sys


# 读取 灵巧手数据 说明
#   可以通过 get_hand_full_info("hand_a") 或 get_hand_full_info("hand_b") 读取两只手的 所有数据，包括 位置信息，电流数据，扭矩数据，帕悉尼数据，温度数据等。
# 保存数据
def save_as_bson(data, filename="xhand_control_data.bson"):
    # 将numpy数组转换为可序列化的list
    serializable_data = {
        'left_hand': data['left_hand'].tolist(),
        'right_hand': data['right_hand'].tolist()
    }
    
    # 编码为BSON并保存
    with open(filename, 'wb') as f:
        f.write(BSON.encode(serializable_data))
    print(f"控制数据已保存至 {filename}")

def load_bson(bson_file: str) -> dict:
    with open(bson_file, "rb") as f:
        data = BSON.decode(f.read())
    print(f"Loaded BSON data from {bson_file}")
    return data

if __name__ == "__main__":
    freq = 20
    node = XHandTeleOps("config_without_xhand.yaml")

    # 获取 左右手
    resp_ht = node.get_hand_type("hand_a")
    if resp_ht and resp_ht["code"] == 0:
        hand_a_type = resp_ht['data']
        print(f"hand_a_type: {hand_a_type}")

    # 获取 SN 码
    resp_sn = node.get_serial_number("hand_a")
    if resp_sn and resp_sn["code"] == 0:
        hand_a_sn = resp_sn['data']
        print(f"hand_a_sn: {hand_a_sn}")


    # 灵巧手回放使用的 BSON 路径。默认指向项目自带的 samples/。
    # 迁移时如需换数据，直接改这一行即可。
    _bson_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "samples", "xhand_control_data.bson")
    )
    bson_data = load_bson(_bson_path)
    frames = bson_data["frames"]

    print("初始化完成，等待主控信号...")
    print("READY", flush=True)

    signal = sys.stdin.readline().strip()
    if signal.upper() == "START":
        print("[xhand] 收到 START，开始执行任务")
        start_time = time.perf_counter()
        for frame in frames:
            start = time.perf_counter()
            # 等待正确的时间点
            while time.perf_counter() - start_time < frame["t"]:
                pass
            
            # 发送控制指令
            node.send_data_xhand({
                "left_hand": np.array(frame["action"]["left_hand"]),
                "right_hand": np.array(frame["action"]["right_hand"])
            })
            # time.sleep(max(0, 1 / freq - (time.perf_counter() - start)))
    else:
        print("❌ 未收到有效启动信号，退出")



