#!/usr/bin/env python3
"""
测试修复后的IK算法 - test_ik_fix.py

比较修改前后的IK求解结果，验证修复效果
"""

import numpy as np
import math

# 模拟roboticstoolbox和spatialmath
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
    RTB_AVAILABLE = True
except ImportError:
    print("⚠️ roboticstoolbox或spatialmath未安装，仅进行参数对比")
    RTB_AVAILABLE = False

# 修正前的DH参数 (8-DH-IK.py原始版本)
OLD_L1 = 0.11257  
OLD_L2 = 0.1349   
OLD_L3 = 0.15     
OLD_L4 = 0.0      

# 修正后的DH参数 (参考keyboard_control.py)
NEW_L1 = 0.0304   # 第一连杆长度 (30.4mm)
NEW_L2 = 0.116    # 第二连杆长度 (116mm)  
NEW_L3 = 0.1347   # 第三连杆长度 (134.7mm)
NEW_L4 = 0.15     # 第四连杆长度 (150mm)
NEW_D1 = 0.0542   # 第一关节高度偏移 (54.2mm)

def create_old_robot_model():
    """创建修正前的机器人模型"""
    if not RTB_AVAILABLE:
        return None
        
    robot = rtb.DHRobot([
        rtb.RevoluteDH(a=OLD_L1, alpha=np.pi/2, d=0, offset=0),      
        rtb.RevoluteDH(a=OLD_L2, alpha=0, d=0, offset=np.pi/2),      
        rtb.RevoluteDH(a=OLD_L3, alpha=0, d=0, offset=-np.pi/2),     
        rtb.RevoluteDH(a=OLD_L4, alpha=0, d=0, offset=0)             
    ], name="Old SO100 Robot")
    
    return robot

def create_new_robot_model():
    """创建修正后的机器人模型"""
    if not RTB_AVAILABLE:
        return None
        
    robot = rtb.DHRobot([
        rtb.RevoluteDH(a=NEW_L1, alpha=np.pi/2, d=NEW_D1, offset=0),      
        rtb.RevoluteDH(a=NEW_L2, alpha=0, d=0, offset=np.pi/2),       
        rtb.RevoluteDH(a=NEW_L3, alpha=0, d=0, offset=-np.pi/2),      
        rtb.RevoluteDH(a=NEW_L4, alpha=0, d=0, offset=0)              
    ], name="New SO100 Robot")
    
    return robot

def convert_angles_to_robot(q_solver):
    """将求解器角度转换为实际机器人角度"""
    return [-angle for angle in q_solver]

def convert_angles_from_robot(q_robot):
    """将实际机器人角度转换为求解器角度"""
    return [-angle for angle in q_robot]

def old_solve_ik(robot, target_pos, current_q=[0,0,0,0]):
    """修正前的IK求解方法"""
    if robot is None:
        return None, False
        
    try:
        # 错误的变换矩阵构建方法
        T_target = SE3(target_pos[0], target_pos[1], target_pos[2])
        
        # 错误的角度转换
        ik_initial = np.array(current_q[:4])
        ik_initial[1] = -ik_initial[1]  # Pitch取反
        ik_initial[2] = -ik_initial[2]  # Elbow取反
        
        sol = robot.ikine_LM(
            T_target, 
            q0=ik_initial, 
            mask=[1,1,1,0,0,0],
            tol=1e-4, 
            joint_limits=True,
            slimit=20
        )
        
        if sol.success:
            ik_solution = sol.q
            mujoco_solution = ik_solution.copy()
            mujoco_solution[1] = -mujoco_solution[1]  # Pitch取反
            mujoco_solution[2] = -mujoco_solution[2]  # Elbow取反
            return mujoco_solution, True
        else:
            return None, False
            
    except Exception as e:
        print(f"老方法IK失败: {e}")
        return None, False

def new_solve_ik(robot, target_pos, current_q=[0,0,0,0]):
    """修正后的IK求解方法"""
    if robot is None:
        return None, False
        
    try:
        # 正确的变换矩阵构建方法
        T_target = SE3.Tx(target_pos[0]) * SE3.Ty(target_pos[1]) * SE3.Tz(target_pos[2])
        
        # 正确的角度转换
        current_q_solver = convert_angles_from_robot(current_q)
        
        sol = robot.ikine_LM(
            T_target, 
            q0=current_q_solver, 
            mask=[1,1,1,0,0,0],
            tol=1e-4, 
            joint_limits=True,
            slimit=20
        )
        
        if sol.success:
            q_solution = convert_angles_to_robot(sol.q.tolist())
            return np.array(q_solution), True
        else:
            return None, False
            
    except Exception as e:
        print(f"新方法IK失败: {e}")
        return None, False

def test_ik_comparison():
    """比较修正前后的IK求解效果"""
    print("🔧 IK算法修正效果对比测试")
    print("=" * 60)
    
    # 参数对比
    print("📊 DH参数对比:")
    print(f"   L1: {OLD_L1:.6f} → {NEW_L1:.6f} ({NEW_L1-OLD_L1:+.6f})")
    print(f"   L2: {OLD_L2:.6f} → {NEW_L2:.6f} ({NEW_L2-OLD_L2:+.6f})")
    print(f"   L3: {OLD_L3:.6f} → {NEW_L3:.6f} ({NEW_L3-OLD_L3:+.6f})")
    print(f"   L4: {OLD_L4:.6f} → {NEW_L4:.6f} ({NEW_L4-OLD_L4:+.6f})")
    print(f"   D1: 0.000000 → {NEW_D1:.6f} (+{NEW_D1:.6f})")
    
    if not RTB_AVAILABLE:
        print("\n⚠️ 无法进行实际IK测试，请安装roboticstoolbox-python")
        return
    
    # 创建模型
    old_robot = create_old_robot_model()
    new_robot = create_new_robot_model()
    
    # 测试位置
    test_positions = [
        ([0.2, 0.0, 0.1], "中心位置"),
        ([0.25, 0.0, 0.05], "红色cube位置"),
        ([0.15, 0.1, 0.12], "右侧位置"),
        ([0.18, -0.08, 0.08], "左侧位置"),
        ([0.22, 0.0, 0.15], "较高位置"),
    ]
    
    print(f"\n🎯 IK求解效果对比:")
    print("-" * 60)
    
    success_old = 0
    success_new = 0
    
    for i, (pos, desc) in enumerate(test_positions):
        print(f"\n测试 {i+1}: {desc}")
        print(f"目标位置: X={pos[0]*1000:.0f}mm, Y={pos[1]*1000:.0f}mm, Z={pos[2]*1000:.0f}mm")
        
        # 测试旧方法
        old_result, old_success = old_solve_ik(old_robot, pos)
        if old_success:
            success_old += 1
            old_angles_deg = [math.degrees(a) for a in old_result]
            print(f"  旧方法: ✅ [{', '.join([f'{a:.1f}°' for a in old_angles_deg])}]")
        else:
            print(f"  旧方法: ❌ 求解失败")
        
        # 测试新方法
        new_result, new_success = new_solve_ik(new_robot, pos)
        if new_success:
            success_new += 1
            new_angles_deg = [math.degrees(a) for a in new_result]
            print(f"  新方法: ✅ [{', '.join([f'{a:.1f}°' for a in new_angles_deg])}]")
            
            # 验证正向运动学
            T_fk = new_robot.fkine(convert_angles_from_robot(new_result))
            actual_pos = T_fk.t
            error = np.linalg.norm(np.array(pos) - actual_pos)
            print(f"  验证FK: 误差={error*1000:.2f}mm")
        else:
            print(f"  新方法: ❌ 求解失败")
    
    print(f"\n📈 总结:")
    print(f"  旧方法成功率: {success_old}/{len(test_positions)} ({success_old/len(test_positions)*100:.0f}%)")
    print(f"  新方法成功率: {success_new}/{len(test_positions)} ({success_new/len(test_positions)*100:.0f}%)")
    
    if success_new > success_old:
        print(f"  🎉 修正后IK性能提升: +{success_new-success_old}个测试点")
    elif success_new == success_old:
        print(f"  ✅ 修正后IK性能保持稳定")
    else:
        print(f"  ⚠️ 修正后IK性能下降: -{success_old-success_new}个测试点")

def test_home_position():
    """测试Home位置的正向运动学"""
    print(f"\n🏠 Home位置验证:")
    print("-" * 30)
    
    if not RTB_AVAILABLE:
        print("⚠️ 无法进行测试")
        return
    
    # Home角度: [0°, -90°, 90°, 0°]
    home_angles = [0, math.radians(-90), math.radians(90), 0]
    
    # 旧模型
    old_robot = create_old_robot_model()
    old_T = old_robot.fkine(home_angles)
    old_pos = old_T.t
    
    # 新模型  
    new_robot = create_new_robot_model()
    new_T = new_robot.fkine(home_angles)
    new_pos = new_T.t
    
    print(f"旧模型Home位置: X={old_pos[0]*1000:.1f}mm, Y={old_pos[1]*1000:.1f}mm, Z={old_pos[2]*1000:.1f}mm")
    print(f"新模型Home位置: X={new_pos[0]*1000:.1f}mm, Y={new_pos[1]*1000:.1f}mm, Z={new_pos[2]*1000:.1f}mm")
    
    # 计算差异
    diff = new_pos - old_pos
    print(f"位置差异: ΔX={diff[0]*1000:+.1f}mm, ΔY={diff[1]*1000:+.1f}mm, ΔZ={diff[2]*1000:+.1f}mm")

if __name__ == "__main__":
    test_home_position()
    test_ik_comparison() 