import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import threading

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Load model
xml_path = "scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# 获取末端执行器ID
end_effector_id = mjmodel.body("Moving_Jaw").id
print(f"✅ 末端执行器: Moving_Jaw (ID: {end_effector_id})")

# ========== 自动获取连杆长度 ==========
def get_link_lengths_from_mjmodel(mjmodel):
    # 通过body的pos属性累加获取连杆长度，单位为米
    # 结构: Base -> Rotation_Pitch -> Upper_Arm -> Lower_Arm -> Wrist_Pitch_Roll -> ...
    # 只取y和z分量（机械臂在y-z平面展开）
    # 1. Base到Upper_Arm（Upper_Arm相对Rotation_Pitch）
    pos1 = mjmodel.body('Upper_Arm').pos # 相对父节点
    l1 = (pos1[1]**2 + pos1[2]**2) ** 0.5
    # 2. Upper_Arm到Lower_Arm
    pos2 = mjmodel.body('Lower_Arm').pos
    l2 = (pos2[1]**2 + pos2[2]**2) ** 0.5
    # 3. Lower_Arm到Wrist_Pitch_Roll
    pos3 = mjmodel.body('Wrist_Pitch_Roll').pos
    l3 = (pos3[1]**2 + pos3[2]**2) ** 0.5
    # 4. Wrist_Pitch_Roll到末端（Fixed_Jaw/Moving_Jaw）
    pos4 = mjmodel.body('Fixed_Jaw').pos
    l4 = (pos4[1]**2 + pos4[2]**2) ** 0.5
    # 末端偏移（Moving_Jaw相对Fixed_Jaw）
    pos5 = mjmodel.body('Moving_Jaw').pos
    l5 = (pos5[0]**2 + pos5[1]**2 + pos5[2]**2) ** 0.5
    return l1, l2, l3, l4, l5

# 获取连杆长度
L1, L2, L3, L4, L5 = get_link_lengths_from_mjmodel(mjmodel)
print(f"自动获取连杆长度: L1={L1:.4f} L2={L2:.4f} L3={L3:.4f} L4={L4:.4f} L5={L5:.4f}")

# ===================== 关节方向修正 =====================
# 1,2,3号关节（Pitch, Elbow, Wrist_Pitch）正方向为顺时针（即数学负方向），需在IK和正解中取反
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

# ===================== IK算法 =====================
def inverse_kinematics(x, y, l1=L1, l2=L2):
    """2D IK算法，修正关节正方向，自动用模型连杆长度"""
    theta1_offset = -math.atan2(0.028, 0.11257)
    theta2_offset = -math.atan2(0.0052, 0.1349) + theta1_offset
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.pi - math.acos(cos_theta2)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma
    # 关节2、3正方向修正（顺时针为正，取反）
    joint2 = -(theta1 - theta1_offset)
    joint3 = -(theta2 - theta2_offset)
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    return joint2, joint3

def spatial_inverse_kinematics(target_x, target_y, target_z):
    """
    空间IK，修正关节正方向
    """
    try:
        # 1. 基座旋转角度
        base_rotation = math.atan2(target_y, target_x)
        # 2. 水平距离
        r_horizontal = math.sqrt(target_x**2 + target_y**2)
        # 3. 用(r, z)做二维IK
        joint2, joint3 = inverse_kinematics(r_horizontal, target_z)
        # 4. Wrist_Pitch正方向修正（顺时针为正，取反）
        wrist_pitch = -( -(joint2 + joint3) )  # 先求-(joint2+joint3)，再取反
        # 5. 关节限制
        base_rotation = max(-math.pi, min(math.pi, base_rotation))
        wrist_pitch = max(-math.pi, min(math.pi, wrist_pitch))
        joint_angles = np.array([base_rotation, joint2, joint3, wrist_pitch])
        print(f"🔧 IK: 目标[{target_x:.3f}, {target_y:.3f}, {target_z:.3f}] → 角度[{', '.join([f'{math.degrees(a):.1f}°' for a in joint_angles])}]")
        return joint_angles, True
    except Exception as e:
        print(f"❌ IK失败: {e}")
        return None, False

# ===================== 正向运动学 =====================
def get_end_effector_position_from_joints():
    """通过正向运动学计算末端执行器位置（更可靠的方法）"""
    current_angles = mjdata.qpos[qpos_indices]
    base_rot = current_angles[0]
    pitch = -current_angles[1]  # 取反
    elbow = -current_angles[2]  # 取反
    l1, l2 = L1, L2
    theta1_offset = -math.atan2(0.028, 0.11257)
    theta2_offset = -math.atan2(0.0052, 0.1349) + theta1_offset
    theta1 = pitch + theta1_offset
    theta2 = elbow + theta2_offset
    x_2d = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    z_2d = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    x_3d = x_2d * math.cos(base_rot)
    y_3d = x_2d * math.sin(base_rot)
    z_3d = z_2d
    return np.array([x_3d, y_3d, z_3d])

# ===================== 初始化home位 =====================
def main():
    # 初始化home位
    mjdata.qpos[qpos_indices] = HOME_QPOS
    mujoco.mj_forward(mjmodel, mjdata)
    print("🚀 优化的IK测试系统")
    print("=" * 40)
    initial_pos = get_end_effector_position_from_joints()
    print(f"📍 初始位置: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
    
    try:
        with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
            # 启动输入线程
            input_thread_handle = threading.Thread(target=input_thread, daemon=True)
            input_thread_handle.start()
            
            print("\n✅ MuJoCo查看器已启动")
            
            # 主循环
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 测试结束")
    
    print("👋 系统已停止")

def input_thread():
    """用户输入处理"""
    print("\n📝 输入格式: x y z")
    print("📝 输入 'q' 退出")
    print("📝 输入 'test' 运行测试序列")
    print("=" * 40)
    
    while True:
        try:
            user_input = input("🔢 坐标: ").strip()
            
            if user_input.lower() == 'q':
                print("👋 退出")
                os._exit(0)
            
            if user_input.lower() == 'test':
                # 运行测试序列
                test_positions = [
                    (0.15, 0.0, 0.1),
                    (0.1, 0.1, 0.05),
                    (0.1, -0.1, 0.05),
                    (0.2, 0.0, 0.0),
                    (0.0, 0.15, 0.1),
                ]
                for i, (x, y, z) in enumerate(test_positions):
                    print(f"\n🧪 测试 {i+1}:")
                    move_to_position(x, y, z)
                    time.sleep(1)
                continue
                
            # 解析坐标
            coords = user_input.split()
            if len(coords) != 3:
                print("❌ 请输入3个坐标值")
                continue
                
            x, y, z = map(float, coords)
            move_to_position(x, y, z)
            
        except ValueError:
            print("❌ 请输入有效数字")
        except KeyboardInterrupt:
            print("\n👋 退出")
            os._exit(0)
        except Exception as e:
            print(f"❌ 错误: {e}")

# ===================== 移动到指定位置 =====================
def move_to_position(target_x, target_y, target_z):
    """移动到指定位置"""
    print(f"\n🎯 目标: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")
    # 计算IK
    joint_angles, success = spatial_inverse_kinematics(target_x, target_y, target_z)
    if success and joint_angles is not None:
        # 应用关节角度
        mjdata.qpos[qpos_indices[0]] = joint_angles[0]  # Rotation
        mjdata.qpos[qpos_indices[1]] = joint_angles[1]  # Pitch
        mjdata.qpos[qpos_indices[2]] = joint_angles[2]  # Elbow
        mjdata.qpos[qpos_indices[3]] = joint_angles[3]  # Wrist_Pitch
        mujoco.mj_forward(mjmodel, mjdata)
        # 使用正向运动学验证位置
        calculated_pos = get_end_effector_position_from_joints()
        error = np.linalg.norm([target_x, target_y, target_z] - calculated_pos)
        print(f"✅ 计算位置: [{calculated_pos[0]:.3f}, {calculated_pos[1]:.3f}, {calculated_pos[2]:.3f}]")
        print(f"📏 误差: {error:.4f}m")
        if error < 0.01:
            print("✅ 精度很高")
        elif error < 0.05:
            print("⚠️  精度中等")
        else:
            print("❌ 精度较低")
        print("-" * 40)
        return True
    else:
        print("❌ IK求解失败")
        print("-" * 40)
        return False

if __name__ == "__main__":
    main()