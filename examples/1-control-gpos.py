import os
import mujoco
import mujoco.viewer
import numpy as np
import time

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

# Initialize target joint positions
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()  # Copy the initial joint positions
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()

# Thread-safe lock
lock = threading.Lock()

# 添加变量来跟踪位置变化和显示控制
last_displayed_gpos = target_gpos.copy()
position_changed = False
show_status_counter = 0  # 用于定期显示状态
DISPLAY_INTERVAL = 100   # 每100个循环显示一次状态（如果没有变化）

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

# 添加函数来检查位置是否发生变化
def positions_changed(pos1, pos2, tolerance=1e-3):
    """检查两个位置数组是否发生了显著变化"""
    return np.any(np.abs(pos1 - pos2) > tolerance)

# Handle key press events
def on_press(key):
    global position_changed
    try:
        k = key.char.lower()  # Convert to lowercase to handle both upper and lower case inputs
        if k in key_to_joint_increase:
            with lock:
                keys_pressed[k] = 1  # Increase direction
                position_changed = True
        elif k in key_to_joint_decrease:
            with lock:
                keys_pressed[k] = -1  # Decrease direction  
                position_changed = True
        elif k == "0":
            with lock:
                global target_qpos, target_gpos
                target_qpos = init_qpos.copy()  # Reset to initial position
                target_gpos = init_gpos.copy()  # Reset to initial gripper position
                position_changed = True
            print("🔄 Reset to initial position")
        
        # 只在有效按键时显示
        if k in key_to_joint_increase or k in key_to_joint_decrease or k == "0":
            key_name = key.char if hasattr(key, 'char') else str(key)
            print(f"⌨️  Key pressed: {key_name}")

    except AttributeError:
        pass  # Handle special keys if necessary

# Handle key release events
def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
                # 按键释放时也显示一次当前状态
                global position_changed
                position_changed = True
    except AttributeError:
        pass  # Handle special keys if necessary

# Start the keyboard listener in a separate thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Backup for target_gpos in case of invalid IK
target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

print("🚀 Robot control started!")
print("📋 Controls:")
print("   W/S: Forward/Backward    A/D: Left/Right    R/F: Up/Down")
print("   Q/E: Roll +/-           G/T: Pitch +/-     Z/C: Gripper +/-")
print("   0: Reset to initial position")
print("   Ctrl+C: Exit")
print("=" * 60)

try:
    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            
            # 检查是否有按键被按下，如果有则处理位置更新
            movement_occurred = False
            
            with lock:
                if keys_pressed:  # 只有在有按键按下时才处理
                    for k, direction in keys_pressed.items():
                        if k in key_to_joint_increase:
                            position_idx = key_to_joint_increase[k]
                            if position_idx == 1 or position_idx == 5:  # Special handling for joint 1 and 5
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

            # 只在位置发生变化时显示信息
            should_display = False
            
            if movement_occurred or position_changed:
                should_display = True
                position_changed = False
            elif positions_changed(target_gpos, last_displayed_gpos):
                should_display = True
            
            # 定期显示状态（降低频率）
            show_status_counter += 1
            if show_status_counter >= DISPLAY_INTERVAL and not should_display:
                should_display = True
                show_status_counter = 0
                print("📍 Status check - Robot running normally")
            
            if should_display:
                print(f"🎯 target_gpos: [{', '.join([f'{x:.3f}' for x in target_gpos])}]")
                if keys_pressed:
                    active_keys = ', '.join(keys_pressed.keys())
                    print(f"🔄 Active keys: {active_keys}")
                last_displayed_gpos = target_gpos.copy()
                show_status_counter = 0

            fd_qpos = mjdata.qpos[qpos_indices][1:5]
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
            
            if ik_success:  # Check if IK solution is valid
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
                mjdata.qpos[qpos_indices] = target_qpos

                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                # backup
                target_gpos_last = target_gpos.copy()  # Save backup of target_gpos
                target_qpos_last = target_qpos.copy()  # Save backup of target_gpos
            else:
                target_gpos = target_gpos_last.copy()  # Restore the last valid target_gpos
                if should_display:
                    print("⚠️  IK solution failed, restoring last valid position")

            # Time management to maintain simulation timestep
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("\n🛑 User interrupted the simulation.")
finally:
    listener.stop()  # Stop the keyboard listener
    print("👋 Robot control stopped. Goodbye!")
    # viewer.close()  # MuJoCo viewer会自动关闭
