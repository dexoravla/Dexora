
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
from pynput import keyboard
from threading import Event
import sys

# 读取 灵巧手数据 说明
#   可以通过 get_hand_full_info("hand_a") 或 get_hand_full_info("hand_b") 读取两只手的 所有数据，包括 位置信息，电流数据，扭矩数据，帕悉尼数据，温度数据等。
# 保存数据
def save_as_bson(data, filename="xhand_control_data.bson"):    
    # 编码为BSON并保存
    with open(filename, 'wb') as f:
        f.write(BSON.encode({"frames": frames}))
    print(f"控制数据已保存至 {filename}")

class HandKeyboardHandler:
    def __init__(self):
        self.record_event: Event = Event()
        self.exit_flag: bool = False
        self.save_flag: bool = False
    def wait_for_start(self):
        print("📥 请按下空格键开始录制手部动作...")
        self.record_event.wait()  # 阻塞，直到按空格键
        print("✅ 已收到空格键，开始录制...")

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                if not self.record_event.is_set():
                    self.record_event.set()
            elif hasattr(key, 'char') and key.char == 's':
                print("💾 收到 's'，保存 BSON 并结束录制")
                self.save_flag = True
                self.record_event.set()
            elif key == keyboard.Key.esc:
                print("❌ 按下 ESC，终止程序")
                self.exit_flag = True
                self.record_event.set()
                return False  # 停止监听器
        except Exception as e:
            print(f"⚠️ 键盘事件处理出错: {e}")

if __name__ == "__main__":
    node = XHandTeleOps("config.yaml")

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

    # # 重置指尖传感器
    # resp_rs = node.reset_all_sensors("hand_a")
    # if resp_rs and resp_rs["code"] == 200:
    #     print(f"Reset hand_a all sensors successfully")
    handler = HandKeyboardHandler()
    listener = keyboard.Listener(on_press=handler.on_press)
    listener.start()
    handler.wait_for_start()
    if handler.exit_flag:
        print("⛔ 程序提前终止。")
        sys.exit(0)

    try:
        frames = []
        fps = 20
        frame_interval = 1.0 / fps  # 50ms
        frame_idx = 0
        
        # 初始化精确时间控制
        start_time = time.perf_counter()
        next_frame_time = start_time
        
        while True:   
            # 等待到下一帧的精确时间
            current_time = time.perf_counter()
            if current_time < next_frame_time:
                sleep_time = next_frame_time - current_time
                time.sleep(sleep_time)
            
            # 记录帧开始时间戳
            frame_start = time.perf_counter()
            t = int(time.time() * 1000)  # Unix时间戳（毫秒）
            
            # 监控采样频率（每100帧输出一次）
            if frame_idx % 100 == 0:
                elapsed_time = frame_start - start_time
                actual_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
                print(f"📊 当前采样频率: {actual_fps:.2f} Hz (目标: {fps} Hz)")
            # 读取手部数据，示例代码
            # print("\n\n")
            # print("//================================")
            # print("//Read various hand states")
            # print("//================================")
            # 如果 同时 读写，需要 设置 force_update=False；如果只读不写，需要 设置 force_update=True
            resp = node.get_hand_full_info("hand_a", force_update=False, is_print=False)
            if resp and resp["code"] == 200:
                result = resp['data']
                # print(f"关节位置 result['joint_position_dic']: {result['joint_position_dic']}")
                left_hand_pos = [result['joint_position_dic'][f'joint{i}'] for i in range(12)]
                # print("observation:",f"left_hand_pos: {left_hand_pos}")

            resp = node.get_hand_full_info("hand_b", force_update=False, is_print=False)
            if resp and resp["code"] == 200:
                result = resp['data']
                # print(f"关节位置 result['joint_position_dic']: {result['joint_position_dic']}")
                right_hand_pos = [result['joint_position_dic'][f'joint{i}'] for i in range(12)] 
                # print(f"right_hand_pos: {right_hand_pos}")

            # 从 visionpro 读数据
            data = node.get_data_from_visionpro()

            # 转换 + 存储
            transform_data = node.retarget_data(data)
            # print("action:",transform_data["left_hand"])
            # === 实际帧起始时间 ===
            # frame_start = time.perf_counter()
            # t = frame_start - start_time  # 相对时间戳
            frames.append({
                "t": t,
                "action":{
                    "left_hand": transform_data["left_hand"].tolist(),
                    "right_hand": transform_data["right_hand"].tolist()
                },
                "observation":{
                    "left_hand": left_hand_pos,
                    "right_hand": right_hand_pos
                }

            })
            node.send_data_xhand(transform_data)
            
            # 计算下一帧的精确时间
            next_frame_time = start_time + (frame_idx + 1) * frame_interval
            frame_idx += 1
            if handler.save_flag:
                print("📝 开始保存 BSON 文件...")
                save_as_bson(frames)
                print("✅ 数据保存完成，程序结束。")
                break
    except Exception as e:
        print(f"⚠️ 录制时发生异常: {e}")


