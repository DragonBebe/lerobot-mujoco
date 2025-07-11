#!/usr/bin/env python3
"""
关节映射调试脚本 - debug_joint_mapping.py

验证关节角度设置是否正确映射到MuJoCo
"""

import os
import mujoco
import numpy as np
import math

os.environ["MUJOCO_GL"] = "egl"

# 关节名称
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

def test_joint_mapping():
    """测试关节映射"""
    print("🔧 关节映射调试")
    print("=" * 50)
    
    # 加载模型
    mjmodel = mujoco.MjModel.from_xml_path("scene.xml")
    mjdata = mujoco.MjData(mjmodel)
    
    # 获取关节索引
    qpos_indices = np.array([
        mjmodel.jnt_qposadr[mjmodel.joint(name).id] 
        for name in JOINT_NAMES
    ])
    
    # 获取末端执行器ID
    end_effector_id = mjmodel.body("Moving_Jaw").id
    
    print(f"📍 关节索引: {qpos_indices}")
    print(f"📍 末端执行器ID: {end_effector_id}")
    
    # 测试用例：逐个关节移动
    test_cases = [
        ("初始位置 (所有关节0度)", [0, 0, 0, 0]),
        ("关节1旋转90度", [math.pi/2, 0, 0, 0]),
        ("关节2倾斜45度", [0, math.pi/4, 0, 0]),
        ("关节3弯曲45度", [0, 0, math.pi/4, 0]),
        ("关节4腕部45度", [0, 0, 0, math.pi/4]),
        ("Home位置", [0, -math.pi/2, math.pi/2, 0]),
    ]
    
    for desc, angles in test_cases:
        print(f"\n🧪 测试: {desc}")
        print(f"   设置角度: [{', '.join([f'{math.degrees(a):.1f}°' for a in angles])}]")
        
        # 重置所有关节
        mjdata.qpos[:] = 0
        
        # 设置前4个关节角度
        mjdata.qpos[qpos_indices[:4]] = angles
        
        # 更新模型状态
        mujoco.mj_forward(mjmodel, mjdata)
        
        # 获取实际位置
        ee_pos = mjdata.xpos[end_effector_id]
        print(f"   末端位置: X={ee_pos[0]*1000:.1f}mm, Y={ee_pos[1]*1000:.1f}mm, Z={ee_pos[2]*1000:.1f}mm")
        
        # 打印所有关节的实际值
        actual_angles = mjdata.qpos[qpos_indices]
        print(f"   实际角度: [{', '.join([f'{math.degrees(a):.1f}°' for a in actual_angles[:4]])}]")
        
        print("-" * 30)

def compare_with_roboticstoolbox():
    """对比roboticstoolbox的计算结果"""
    print("\n🔬 对比roboticstoolbox计算")
    print("=" * 50)
    
    try:
        import roboticstoolbox as rtb
        from spatialmath import SE3
        
        # 创建相同的DH模型
        L1, L2, L3, L4, D1 = 0.0304, 0.116, 0.1347, 0.15, 0.0542
        
        robot_model = rtb.DHRobot([
            rtb.RevoluteDH(a=L1, alpha=np.pi/2, d=D1, offset=0),
            rtb.RevoluteDH(a=L2, alpha=0, d=0, offset=np.pi/2),
            rtb.RevoluteDH(a=L3, alpha=0, d=0, offset=-np.pi/2),
            rtb.RevoluteDH(a=L4, alpha=0, d=0, offset=0)
        ], name="SO100 Robot")
        
        # 测试角度
        test_angles = [0, -math.pi/2, math.pi/2, 0]  # Home位置
        
        print(f"🔧 测试角度: [{', '.join([f'{math.degrees(a):.1f}°' for a in test_angles])}]")
        
        # roboticstoolbox正运动学
        q_solver = [-angle for angle in test_angles]  # 转换为求解器角度
        T = robot_model.fkine(q_solver)
        rtb_pos = T.t
        
        # 应用坐标系转换
        corrected_pos = [rtb_pos[0], -rtb_pos[1], rtb_pos[2]]
        
        print(f"📊 roboticstoolbox FK: X={corrected_pos[0]*1000:.1f}mm, Y={corrected_pos[1]*1000:.1f}mm, Z={corrected_pos[2]*1000:.1f}mm")
        
        # MuJoCo验证
        mjmodel = mujoco.MjModel.from_xml_path("scene.xml")
        mjdata = mujoco.MjData(mjmodel)
        
        qpos_indices = np.array([
            mjmodel.jnt_qposadr[mjmodel.joint(name).id] 
            for name in JOINT_NAMES
        ])
        end_effector_id = mjmodel.body("Moving_Jaw").id
        
        mjdata.qpos[qpos_indices[:4]] = test_angles
        mujoco.mj_forward(mjmodel, mjdata)
        mujoco_pos = mjdata.xpos[end_effector_id]
        
        print(f"📊 MuJoCo 实际位置: X={mujoco_pos[0]*1000:.1f}mm, Y={mujoco_pos[1]*1000:.1f}mm, Z={mujoco_pos[2]*1000:.1f}mm")
        
        # 计算差异
        diff = np.array(corrected_pos) - mujoco_pos
        error = np.linalg.norm(diff)
        
        print(f"📏 差异: ΔX={diff[0]*1000:+.1f}mm, ΔY={diff[1]*1000:+.1f}mm, ΔZ={diff[2]*1000:+.1f}mm")
        print(f"📐 总误差: {error*1000:.1f}mm")
        
        if error < 0.005:
            print("✅ 误差很小，模型匹配良好")
        elif error < 0.02:
            print("⚠️ 误差中等，可能需要微调")
        else:
            print("❌ 误差较大，需要检查模型差异")
        
    except ImportError:
        print("❌ roboticstoolbox未安装，跳过对比")

if __name__ == "__main__":
    test_joint_mapping()
    compare_with_roboticstoolbox() 