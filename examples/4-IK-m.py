import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import threading

np.set_printoptions(linewidth=200)

# Set up the MuJoCo render backend
os.environ["MUJOCO_GL"] = "egl"

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Load model
xml_path = "scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# 获取末端执行器ID和基座ID
end_effector_id = mjmodel.body("Moving_Jaw").id
base_id = mjmodel.body("Base").id  # 假设基座名称为Base
print(f"✅ 末端执行器: Moving_Jaw (ID: {end_effector_id})")
print(f"✅ 基座: Base (ID: {base_id})")

def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets
    
    Parameters:
        x: End effector x coordinate
        y: End effector y coordinate (注意：这里y实际上是z坐标)
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)
        
    Returns:
        joint2, joint3: Joint angles in radians as defined in the URDF file
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = -math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = -math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0
    
    # Calculate distance from origin to target point
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2  # Maximum reachable distance
    
    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max
    
    # If target point is less than minimum workspace (|l1-l2|), scale it
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min
    
    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)
    
    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma
    
    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 - theta1_offset
    joint3 = theta2 - theta2_offset
    
    # Ensure angles are within URDF limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    
    return joint2, joint3

def spatial_inverse_kinematics(target_x, target_y, target_z):
    """
    3D空间坐标转关节角度
    修正坐标系问题：
    1. Z轴方向修正
    2. 坐标系转换
    """
    try:
        # 1. 计算基座旋转角度（确定x,y方向）
        base_rotation = math.atan2(target_y, target_x)
        
        # 2. 计算水平距离
        r_horizontal = math.sqrt(target_x**2 + target_y**2)
        
        # 3. Z轴坐标修正 - 根据观察，Z轴方向相反
        corrected_z = -target_z  # 修正Z轴方向
        
        # 4. 使用2D IK算法计算pitch和elbow角度
        # 注意：这里传入的是(水平距离, 修正后的Z坐标)
        joint2, joint3 = inverse_kinematics(r_horizontal, corrected_z)
        
        # 5. 计算腕关节角度（保持末端执行器水平）
        wrist_pitch = -(joint2 + joint3)
        
        # 6. 应用关节限制
        base_rotation = max(-math.pi, min(math.pi, base_rotation))
        wrist_pitch = max(-math.pi, min(math.pi, wrist_pitch))
        
        joint_angles = np.array([base_rotation, joint2, joint3, wrist_pitch])
        
        print(f"🔧 IK计算详情:")
        print(f"   原始目标: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")
        print(f"   修正后Z: {corrected_z:.3f}")
        print(f"   水平距离: {r_horizontal:.3f}")
        print(f"   基座旋转: {math.degrees(base_rotation):.1f}°")
        print(f"   关节角度: [{', '.join([f'{math.degrees(a):.1f}°' for a in joint_angles])}]")
        
        return joint_angles, True
    except Exception as e:
        print(f"❌ IK计算失败: {e}")
        return None, False

def get_current_end_effector_pos():
    """获取当前末端执行器位置（相对于基座的位置）"""
    mujoco.mj_forward(mjmodel, mjdata)
    
    # 获取末端执行器的世界坐标
    end_effector_pos = mjdata.xpos[end_effector_id].copy()
    
    # 获取基座的世界坐标
    try:
        base_pos = mjdata.xpos[base_id].copy()
        # 计算相对位置
        relative_pos = end_effector_pos - base_pos
    except:
        # 如果找不到基座，直接使用世界坐标
        relative_pos = end_effector_pos
        print("⚠️  未找到基座，使用世界坐标")
    
    return relative_pos

def get_multiple_end_effector_positions():
    """获取多个可能的末端执行器位置进行对比"""
    mujoco.mj_forward(mjmodel, mjdata)
    
    # 尝试不同的body作为末端执行器
    candidate_bodies = ["Moving_Jaw", "Jaw", "gripper", "end_effector", "tool"]
    
    positions = {}
    for body_name in candidate_bodies:
        try:
            body_id = mjmodel.body(body_name).id
            pos = mjdata.xpos[body_id].copy()
            positions[body_name] = pos
        except:
            continue
    
    return positions

def move_to_position(target_x, target_y, target_z):
    """移动到指定空间坐标并验证结果"""
    print(f"\n🎯 目标位置: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")
    
    # 记录初始位置
    initial_pos = get_current_end_effector_pos()
    print(f"📍 初始位置: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
    
    # 获取多个候选位置进行对比
    all_positions = get_multiple_end_effector_positions()
    if len(all_positions) > 1:
        print("🔍 所有候选末端执行器位置:")
        for name, pos in all_positions.items():
            print(f"   {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # 计算IK
    joint_angles, success = spatial_inverse_kinematics(target_x, target_y, target_z)
    
    if success and joint_angles is not None:
        # 应用关节角度
        mjdata.qpos[qpos_indices[0]] = joint_angles[0]  # Rotation
        mjdata.qpos[qpos_indices[1]] = joint_angles[1]  # Pitch
        mjdata.qpos[qpos_indices[2]] = joint_angles[2]  # Elbow
        mjdata.qpos[qpos_indices[3]] = joint_angles[3]  # Wrist_Pitch
        # Wrist_Roll 和 Jaw 保持不变
        
        # 更新物理状态
        mujoco.mj_forward(mjmodel, mjdata)
        
        # 获取实际位置
        actual_pos = get_current_end_effector_pos()
        
        # 计算误差（考虑Z轴修正）
        target_corrected = np.array([target_x, target_y, target_z])
        error = np.linalg.norm(target_corrected - actual_pos)
        
        print(f"✅ 实际位置: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
        print(f"📏 位置误差: {error:.4f}m")
        
        # 详细的误差分析
        error_x = abs(target_x - actual_pos[0])
        error_y = abs(target_y - actual_pos[1])
        error_z = abs(target_z - actual_pos[2])
        print(f"📊 分轴误差: X={error_x:.4f}m, Y={error_y:.4f}m, Z={error_z:.4f}m")
        
        # 获取移动后的所有候选位置
        all_positions_after = get_multiple_end_effector_positions()
        if len(all_positions_after) > 1:
            print("🔍 移动后所有候选位置:")
            for name, pos in all_positions_after.items():
                pos_error = np.linalg.norm([target_x, target_y, target_z] - pos)
                print(f"   {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] (误差: {pos_error:.4f}m)")
        
        # 误差评估
        if error > 0.05:
            print("❌ 误差较大，可能需要进一步调整")
        elif error > 0.01:
            print("⚠️  误差中等，基本可用")
        else:
            print("✅ 误差很小，IK计算准确")
        
        print("-" * 50)
        return True
    else:
        print("❌ IK求解失败")
        print("-" * 50)
        return False

def input_thread():
    """处理用户输入的线程"""
    print("\n📝 输入格式: x y z (用空格分隔)")
    print("📝 示例: 0.15 0.0 0.1")
    print("📝 输入 'q' 退出程序")
    print("📝 输入 'info' 查看当前状态")
    print("📝 输入 'debug' 查看所有候选末端执行器位置")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("🔢 请输入目标坐标 (x y z): ").strip()
            
            if user_input.lower() == 'q':
                print("👋 程序退出")
                os._exit(0)
            
            if user_input.lower() == 'info':
                current_pos = get_current_end_effector_pos()
                current_angles = mjdata.qpos[qpos_indices]
                print(f"📍 当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                print(f"🔧 当前角度: [{', '.join([f'{a:.3f}' for a in current_angles])}]")
                continue
            
            if user_input.lower() == 'debug':
                all_positions = get_multiple_end_effector_positions()
                print("🔍 所有候选末端执行器位置:")
                for name, pos in all_positions.items():
                    print(f"   {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                continue
                
            # 解析坐标
            coords = user_input.split()
            if len(coords) != 3:
                print("❌ 请输入3个坐标值 (x y z)")
                continue
                
            x, y, z = map(float, coords)
            
            # 执行移动
            move_to_position(x, y, z)
            
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n👋 程序退出")
            os._exit(0)
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def main():
    # 初始化关节位置
    init_qpos = np.array([0.0, -1.5, 1.5, 0.0, -1.57, -0.157])
    mjdata.qpos[qpos_indices] = init_qpos
    mujoco.mj_forward(mjmodel, mjdata)
    
    print("🚀 修正坐标系的IK测试系统")
    print("=" * 50)
    
    # 显示初始状态的所有候选位置
    print("🔍 初始状态下的所有候选末端执行器位置:")
    all_positions = get_multiple_end_effector_positions()
    for name, pos in all_positions.items():
        print(f"   {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # 预设位置测试（使用较小的测试值）
    print("\n📍 测试较小的预设位置...")
    test_positions = [
        (0.1, 0.0, 0.05),    # 较近的正前方
        (0.08, 0.05, 0.03),  # 右前方
        (0.08, -0.05, 0.03), # 左前方
        (0.12, 0.0, 0.0),    # 水平正前方
    ]
    
    for i, (x, y, z) in enumerate(test_positions):
        print(f"\n🧪 预设测试 {i+1}:")
        move_to_position(x, y, z)
        time.sleep(1)
    
    try:
        with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
            # 启动输入线程
            input_thread_handle = threading.Thread(target=input_thread, daemon=True)
            input_thread_handle.start()
            
            print("\n✅ MuJoCo查看器已启动，可以开始输入坐标测试")
            print("💡 提示: 输入'debug'查看所有候选末端执行器位置")
            
            # 主循环
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 测试结束")
    
    print("👋 IK测试系统已停止")

if __name__ == "__main__":
    main()