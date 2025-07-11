import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import tkinter as tk
from tkinter import simpledialog
import math

from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot
from pynput import keyboard
import threading

np.set_printoptions(linewidth=200)

# Set up the MuJoCo render backend
os.environ["MUJOCO_GL"] = "egl"

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Absolute path of the XML model
xml_path = "scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# 🔧 简化：直接指定末端执行器名称
END_EFFECTOR_NAME = "Moving_Jaw"  # 根据XML文件确定的末端执行器名称
try:
    end_effector_id = mjmodel.body(END_EFFECTOR_NAME).id
    print(f"✅ 末端执行器: {END_EFFECTOR_NAME} (ID: {end_effector_id})")
except:
    # 备选方案：使用Fixed_Jaw或最后一个body
    try:
        end_effector_id = mjmodel.body("Fixed_Jaw").id
        print(f"✅ 使用备选末端执行器: Fixed_Jaw (ID: {end_effector_id})")
    except:
        end_effector_id = mjmodel.nbody - 1
        print(f"⚠️  使用默认末端执行器 ID: {end_effector_id}")

# Define joint control increment (in radians)
JOINT_INCREMENT = 0.005  # Can be adjusted as needed
POSITION_INSERMENT = 0.0008

# create robot
robot = get_robot('so100')

# Define joint limits
control_qlimit = [[-2.1, -3.1, -0.0, -1.375,  -1.57, -0.15], 
                  [ 2.1,  0.0,  3.1,  1.475,   3.1,  1.5]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5], 
                  [0.340,  0.4,  0.23, 2.0,  1.57,  1.5]]

# home位: [0, -90, 90, 0, 0, 0] (单位:度)
def deg2rad(x):
    return x * math.pi / 180

HOME_QPOS = np.array([
    deg2rad(0),      # Rotation
    deg2rad(-90),    # Pitch
    deg2rad(90),     # Elbow
    deg2rad(0),      # Wrist_Pitch
    deg2rad(0),      # Wrist_Roll
    deg2rad(0),      # Jaw
])

# Initialize target joint positions
init_qpos = HOME_QPOS.copy()
target_qpos = init_qpos.copy()  # Copy the initial joint positions
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
if len(init_gpos) == 3:
    init_gpos = np.append(init_gpos, 0.0)  # 默认pitch为0度
target_gpos = init_gpos.copy()

# Thread-safe lock
lock = threading.Lock()

# 显示控制变量
key_press_occurred = False
last_key_time = 0
KEY_DISPLAY_COOLDOWN = 0.1

# 用于存储MuJoCo实际末端位置
actual_ee_position = np.zeros(3)
actual_ee_orientation = np.zeros(4)

def get_mujoco_ee_position():
    """从MuJoCo仿真中获取末端执行器的实际位置和姿态"""
    global actual_ee_position, actual_ee_orientation
    
    # 获取位置 (x, y, z)
    actual_ee_position = mjdata.xpos[end_effector_id].copy()
    
    # 获取姿态 (四元数: w, x, y, z)
    actual_ee_orientation = mjdata.xquat[end_effector_id].copy()
    
    return actual_ee_position, actual_ee_orientation

def quat_to_euler(quat):
    """将四元数转换为欧拉角 (roll, pitch, yaw)"""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

# Define key mappings
key_to_joint_increase = {
    'w': 0,  # Move forward
    'a': 1,  # Move right  
    'r': 2,  # Move up
    'q': 3,  # Roll +
    'g': 4,  # Pitch +
    'z': 5,  # Gripper +
}

key_to_joint_decrease = {
    's': 0,  # Move backward
    'd': 1,  # Move left
    'f': 2,  # Move down
    'e': 3,  # Roll -
    't': 4,  # Pitch -
    'c': 5,  # Gripper -
}

# Dictionary to track the currently pressed keys and their direction
keys_pressed = {}

def display_position_info(key_name=None):
    """显示当前位置信息"""
    # 获取MuJoCo中的实际末端位置
    actual_pos, actual_quat = get_mujoco_ee_position()
    actual_euler = quat_to_euler(actual_quat)
    
    if key_name:
        print(f"⌨️  Key: {key_name}")
    
    print(f"🎯 Target: [{', '.join([f'{x:.3f}' for x in target_gpos[:3]])}] (pitch: {target_gpos[3] if len(target_gpos)>3 else 'N/A'})")
    print(f"🤖 Actual: [{', '.join([f'{x:.3f}' for x in actual_pos])}] (RPY: [{', '.join([f'{x:.3f}' for x in actual_euler])}])")
    
    # 计算位置误差
    position_error = np.linalg.norm(target_gpos[:3] - actual_pos)
    print(f"📏 Position Error: {position_error:.4f}m")
    
    if keys_pressed:
        active_keys = ', '.join(keys_pressed.keys())
        print(f"🔄 Active keys: {active_keys}")
    
    print("-" * 50)

# Handle key press events
def on_press(key):
    global key_press_occurred, last_key_time
    try:
        k = key.char.lower()
        current_time = time.time()
        
        if k in key_to_joint_increase:
            with lock:
                keys_pressed[k] = 1
                if current_time - last_key_time > KEY_DISPLAY_COOLDOWN:
                    key_press_occurred = True
                    last_key_time = current_time
                    display_position_info(k.upper())
                    
        elif k in key_to_joint_decrease:
            with lock:
                keys_pressed[k] = -1
                if current_time - last_key_time > KEY_DISPLAY_COOLDOWN:
                    key_press_occurred = True
                    last_key_time = current_time
                    display_position_info(k.upper())
                    
        elif k == "0":
            with lock:
                global target_qpos, target_gpos
                target_qpos = init_qpos.copy()
                target_gpos = init_gpos.copy()
            print("🔄 Reset to initial position")
            display_position_info("0 (Reset)")
            
        elif k == "p":
            print("📍 Current Status:")
            display_position_info("P (Status)")

        elif k == "i":
            print("📝 请输入目标末端空间坐标和俯仰角：")
            pos = input_target_position()
            if pos is not None:
                with lock:
                    target_gpos[:4] = pos  # 假设 target_gpos 前4位为 x, y, z, pitch
                print(f"🎯 新目标: {pos}")

    except AttributeError:
        pass

# Handle key release events
def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
    except AttributeError:
        pass

# Start the keyboard listener in a separate thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Backup for target_gpos in case of invalid IK
target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

print("🚀 Robot control started!")
print(f"🎯 末端执行器: {END_EFFECTOR_NAME} (ID: {end_effector_id})")
print("📋 Controls:")
print("   W/S: Forward/Backward    A/D: Left/Right    R/F: Up/Down")
print("   Q/E: Roll +/-           G/T: Pitch +/-     Z/C: Gripper +/-")
print("   0: Reset to initial position")
print("   P: Show current status")
print("   Ctrl+C: Exit")
print("=" * 60)

# 显示初始状态
print("📍 Initial Status:")
display_position_info("Initial")

def input_target_position():
    def get_float(prompt):
        while True:
            s = input(prompt)
            if s.strip() == '':
                return None
            try:
                return float(s)
            except Exception:
                print("请输入合法的数字！")
    try:
        x = get_float("请输入目标X坐标（米），直接回车退出：")
        if x is None:
            return None
        y = get_float("请输入目标Y坐标（米），直接回车退出：")
        if y is None:
            return None
        z = get_float("请输入目标Z坐标（米），直接回车退出：")
        if z is None:
            return None
        pitch = get_float("请输入末端俯仰角（度），直接回车退出：")
        if pitch is None:
            return None
        return np.array([x, y, z, pitch])
    except Exception as e:
        print("输入有误，请重新输入。", e)
        return None

try:
    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            
            movement_occurred = False
            
            with lock:
                if keys_pressed:
                    for k, direction in keys_pressed.items():
                        if k in key_to_joint_increase:
                            position_idx = key_to_joint_increase[k]
                            if position_idx == 1 or position_idx == 5:
                                position_idx = 0 if position_idx == 1 else 5
                                if (target_qpos[position_idx]) < control_qlimit[1][position_idx] - JOINT_INCREMENT * direction:
                                    target_qpos[position_idx] += JOINT_INCREMENT * direction
                                    movement_occurred = True
                            elif position_idx == 4 or position_idx == 3:
                                if target_gpos[position_idx] <= control_glimit[1][position_idx]:
                                    target_gpos[position_idx] += POSITION_INSERMENT * direction * 4
                                    movement_occurred = True
                            else:
                                if target_gpos[position_idx] <= control_glimit[1][position_idx]:
                                    target_gpos[position_idx] += POSITION_INSERMENT * direction
                                    movement_occurred = True
                            
                        elif k in key_to_joint_decrease:
                            position_idx = key_to_joint_decrease[k]
                            if position_idx == 1 or position_idx == 5:
                                position_idx = 0 if position_idx == 1 else 5
                                if (target_qpos[position_idx]) > control_qlimit[0][position_idx] - JOINT_INCREMENT * direction:
                                    target_qpos[position_idx] += JOINT_INCREMENT * direction
                                    movement_occurred = True
                            elif position_idx == 4 or position_idx == 3:
                                if target_gpos[position_idx] >= control_glimit[0][position_idx]:
                                    target_gpos[position_idx] += POSITION_INSERMENT * direction * 4
                                    movement_occurred = True
                            else:
                                if target_gpos[position_idx] >= control_glimit[0][position_idx]:
                                    target_gpos[position_idx] += POSITION_INSERMENT * direction
                                    movement_occurred = True

            fd_qpos = mjdata.qpos[qpos_indices][1:5]
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
            
            print(f"qpos_inv: {qpos_inv}, ik_success: {ik_success}")
            
            if ik_success:
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
                mjdata.qpos[qpos_indices] = target_qpos

                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                # backup
                target_gpos_last = target_gpos.copy()
                target_qpos_last = target_qpos.copy()
            else:
                target_gpos = target_gpos_last.copy()
                if key_press_occurred:
                    print("⚠️  IK solution failed, restoring last valid position")

            key_press_occurred = False

            # Time management to maintain simulation timestep
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("\n🛑 User interrupted the simulation.")
finally:
    listener.stop()
    print("👋 Robot control stopped. Goodbye!")
