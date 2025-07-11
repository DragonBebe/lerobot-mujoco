import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import threading

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

# 关节名称
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# 加载模型
xml_path = "scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# 获取末端执行器ID
eef_id = mjmodel.body("Moving_Jaw").id
print(f"✅ 末端执行器: Moving_Jaw (ID: {eef_id})")

def get_end_effector_position():
    """获取末端执行器的世界坐标"""
    return mjdata.xpos[eef_id].copy()

def print_joint_and_eef():
    pos = get_end_effector_position()
    joint_angles = mjdata.qpos[qpos_indices]
    print("当前关节角度（度）:")
    for i, angle in enumerate(joint_angles):
        print(f"  {i}: {JOINT_NAMES[i]:<12} {angle:>7.3f} [deg: {math.degrees(angle):>7.2f}]")
    print(f"末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

def input_thread_func():
    while True:
        user_input = input("请输入指令: ").strip()
        if user_input.lower() == 'q':
            print("👋 退出")
            os._exit(0)
        if user_input.lower() == 'show':
            print_joint_and_eef()
            continue
        if user_input.lower() == 'reset':
            mjdata.qpos[qpos_indices] = 0
            mujoco.mj_forward(mjmodel, mjdata)
            print("已重置所有关节为0.")
            print_joint_and_eef()
            continue
        try:
            parts = user_input.split()
            if len(parts) != 2:
                print("❌ 输入格式错误，应为: 关节序号 角度(度)")
                continue
            idx = int(parts[0])
            deg = float(parts[1])
            if idx < 0 or idx >= len(JOINT_NAMES):
                print(f"❌ 关节序号应在0~{len(JOINT_NAMES)-1}之间")
                continue
            rad = math.radians(deg)
            mjdata.qpos[qpos_indices[idx]] = rad
            mujoco.mj_forward(mjmodel, mjdata)
            print_joint_and_eef()
        except Exception as e:
            print(f"❌ 错误: {e}")

def main():
    print("\n🚀 手动关节控制与末端位置观测工具")
    print("========================================")
    print("输入格式: 关节序号(0-5) 角度(度)，如 1 -90")
    print("输入 show 可显示当前关节和末端位置")
    print("输入 reset 可重置所有关节为0")
    print("输入 q 退出\n")
    try:
        with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
            # 启动输入线程
            t = threading.Thread(target=input_thread_func, daemon=True)
            t.start()
            # 主线程持续刷新viewer
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n🛑 结束")
    print("👋 系统已停止")

if __name__ == "__main__":
    main()