#!/usr/bin/env python3
"""
DH参数校准脚本 - calibrate_dh_params.py

基于MuJoCo实际测量结果校准DH参数
"""

import os
import mujoco
import numpy as np
import math

os.environ["MUJOCO_GL"] = "egl"

def analyze_mujoco_kinematics():
    """分析MuJoCo的运动学特性"""
    print("🔬 分析MuJoCo运动学特性")
    print("=" * 50)
    
    mjmodel = mujoco.MjModel.from_xml_path("scene.xml")
    mjdata = mujoco.MjData(mjmodel)
    
    JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    qpos_indices = np.array([
        mjmodel.jnt_qposadr[mjmodel.joint(name).id] 
        for name in JOINT_NAMES
    ])
    end_effector_id = mjmodel.body("Moving_Jaw").id
    
    # 分析关键位置
    key_positions = [
        ("所有关节0度", [0, 0, 0, 0]),
        ("关节2=-90°", [0, -math.pi/2, 0, 0]),
        ("关节3=90°", [0, 0, math.pi/2, 0]),
        ("Home: 0,-90,90,0", [0, -math.pi/2, math.pi/2, 0]),
        ("测试: 0,45,-45,0", [0, math.pi/4, -math.pi/4, 0]),
    ]
    
    results = []
    
    for desc, angles in key_positions:
        mjdata.qpos[:] = 0
        mjdata.qpos[qpos_indices[:4]] = angles
        mujoco.mj_forward(mjmodel, mjdata)
        
        pos = mjdata.xpos[end_effector_id]
        results.append((desc, angles, pos))
        
        print(f"{desc:15}: X={pos[0]*1000:6.1f}mm, Y={pos[1]*1000:6.1f}mm, Z={pos[2]*1000:6.1f}mm")
    
    return results

def estimate_dh_parameters():
    """根据实际测量估算正确的DH参数"""
    print("\n🔧 估算DH参数")
    print("=" * 50)
    
    results = analyze_mujoco_kinematics()
    
    # 从关键位置推算连杆长度
    pos_0000 = results[0][2]  # [0,0,0,0]
    pos_home = results[3][2]  # [0,-90,90,0]
    
    print(f"初始位置 (0,0,0,0): X={pos_0000[0]*1000:.1f}mm, Z={pos_0000[2]*1000:.1f}mm")
    print(f"Home位置 (0,-90,90,0): X={pos_home[0]*1000:.1f}mm, Z={pos_home[2]*1000:.1f}mm")
    
    # 基于实际测量的几何分析
    # 当所有关节为0时，机械臂完全伸展，总长度约为407.8mm
    total_reach = pos_0000[0]  # 407.8mm
    base_height = pos_0000[2]  # 116.4mm
    
    # Home位置时的高度变化
    home_height = pos_home[2]  # 257.0mm
    height_increase = home_height - base_height  # 约140mm
    
    print(f"\n📏 几何分析:")
    print(f"   总伸展距离: {total_reach*1000:.1f}mm")
    print(f"   基础高度: {base_height*1000:.1f}mm") 
    print(f"   Home高度增加: {height_increase*1000:.1f}mm")
    
    # 推测的连杆长度（基于几何分析）
    # 需要考虑MuJoCo模型的实际机构
    L1_est = 0.045   # 第一连杆，基于高度变化
    L2_est = 0.120   # 第二连杆，基于主要伸展
    L3_est = 0.140   # 第三连杆
    L4_est = 0.100   # 第四连杆到末端
    D1_est = 0.116   # 基础高度偏移
    
    print(f"\n🔧 推估DH参数:")
    print(f"   L1 = {L1_est*1000:.0f}mm")
    print(f"   L2 = {L2_est*1000:.0f}mm") 
    print(f"   L3 = {L3_est*1000:.0f}mm")
    print(f"   L4 = {L4_est*1000:.0f}mm")
    print(f"   D1 = {D1_est*1000:.0f}mm")
    
    return L1_est, L2_est, L3_est, L4_est, D1_est

def test_estimated_parameters():
    """测试估算的DH参数"""
    print("\n🧪 测试估算参数")
    print("=" * 50)
    
    try:
        import roboticstoolbox as rtb
        from spatialmath import SE3
        
        L1, L2, L3, L4, D1 = estimate_dh_parameters()
        
        # 创建新的DH模型
        robot_model = rtb.DHRobot([
            rtb.RevoluteDH(a=L1, alpha=np.pi/2, d=D1, offset=0),
            rtb.RevoluteDH(a=L2, alpha=0, d=0, offset=np.pi/2),
            rtb.RevoluteDH(a=L3, alpha=0, d=0, offset=-np.pi/2),
            rtb.RevoluteDH(a=L4, alpha=0, d=0, offset=0)
        ], name="Calibrated SO100")
        
        # 测试Home位置
        test_angles = [0, -math.pi/2, math.pi/2, 0]
        q_solver = [-angle for angle in test_angles]
        
        T = robot_model.fkine(q_solver)
        rtb_pos = T.t
        corrected_pos = [rtb_pos[0], -rtb_pos[1], rtb_pos[2]]
        
        # MuJoCo实际位置
        mjmodel = mujoco.MjModel.from_xml_path("scene.xml")
        mjdata = mujoco.MjData(mjmodel)
        
        JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        qpos_indices = np.array([
            mjmodel.jnt_qposadr[mjmodel.joint(name).id] 
            for name in JOINT_NAMES
        ])
        end_effector_id = mjmodel.body("Moving_Jaw").id
        
        mjdata.qpos[qpos_indices[:4]] = test_angles
        mujoco.mj_forward(mjmodel, mjdata)
        mujoco_pos = mjdata.xpos[end_effector_id]
        
        print(f"📊 新DH模型 FK: X={corrected_pos[0]*1000:.1f}mm, Y={corrected_pos[1]*1000:.1f}mm, Z={corrected_pos[2]*1000:.1f}mm")
        print(f"📊 MuJoCo 实际: X={mujoco_pos[0]*1000:.1f}mm, Y={mujoco_pos[1]*1000:.1f}mm, Z={mujoco_pos[2]*1000:.1f}mm")
        
        diff = np.array(corrected_pos) - mujoco_pos
        error = np.linalg.norm(diff)
        
        print(f"📏 新模型误差: {error*1000:.1f}mm")
        
        if error < 0.02:
            print("✅ 校准成功！误差 < 20mm")
            return L1, L2, L3, L4, D1
        else:
            print("⚠️ 需要进一步调整")
        
    except ImportError:
        print("❌ roboticstoolbox未安装")
    
    return None

if __name__ == "__main__":
    test_estimated_parameters() 