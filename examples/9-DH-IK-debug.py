#!/usr/bin/env python3
"""
机械臂坐标系偏移调试脚本 - 9-DH-IK-debug.py

🎯 目标：
1. 分析MuJoCo坐标系与FK模型坐标系的偏移
2. 找出系统性的位置偏差
3. 计算准确的坐标变换矩阵

🔧 调试步骤：
1. 测试多个已知关节角度
2. 对比FK计算位置与MuJoCo实际位置
3. 计算固定偏移量
4. 验证偏移修正效果
"""

import os
import mujoco
import numpy as np
import math
from typing import List, Tuple

# 设置环境
np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

# 导入机器人学库
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
    RTB_AVAILABLE = True
except ImportError:
    print("❌ 需要安装 roboticstoolbox-python 和 spatialmath-python")
    RTB_AVAILABLE = False

# ===================== 配置参数 =====================
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# DH参数（与主程序保持一致）
L1 = 0.0304   # 第一连杆长度 (30.4mm)
L2 = 0.116    # 第二连杆长度 (116mm)  
L3 = 0.1347   # 第三连杆长度 (134.7mm)
L4 = 0.15     # 第四连杆长度 (150mm)
D1 = 0.0542   # 第一关节高度偏移 (54.2mm)

# Home位置
HOME_ANGLES_DEG = [0, -90, 90, 0, 0, 0]
HOME_QPOS = np.array([math.radians(angle) for angle in HOME_ANGLES_DEG])

class CoordinateDebugger:
    def __init__(self, xml_path: str):
        """初始化坐标系调试器"""
        self.xml_path = xml_path
        self.mjmodel = None
        self.mjdata = None
        self.qpos_indices = None
        self.end_effector_id = None
        self.robot_model = None
        
        self._init_mujoco()
        if RTB_AVAILABLE:
            self._init_robot_model()
    
    def _init_mujoco(self):
        """初始化MuJoCo模型"""
        print("🔧 初始化MuJoCo模型...")
        self.mjmodel = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mjdata = mujoco.MjData(self.mjmodel)
        
        # 获取关节索引
        self.qpos_indices = np.array([
            self.mjmodel.jnt_qposadr[self.mjmodel.joint(name).id] 
            for name in JOINT_NAMES
        ])
        
        # 获取末端执行器ID
        self.end_effector_id = self.mjmodel.body("Moving_Jaw").id
        
        # 初始化到Home位置
        self.mjdata.qpos[self.qpos_indices] = HOME_QPOS.copy()
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        
        print(f"✅ MuJoCo初始化完成 - 末端执行器ID: {self.end_effector_id}")
    
    def _init_robot_model(self):
        """初始化DH机器人模型"""
        print("🔧 初始化DH机器人模型...")
        self.robot_model = rtb.DHRobot([
            rtb.RevoluteDH(a=L1, alpha=np.pi/2, d=D1, offset=0),      
            rtb.RevoluteDH(a=L2, alpha=0, d=0, offset=np.pi/2),       
            rtb.RevoluteDH(a=L3, alpha=0, d=0, offset=-np.pi/2),      
            rtb.RevoluteDH(a=L4, alpha=0, d=0, offset=0)              
        ], name="SO100 Robot")
        
        print("✅ DH机器人模型创建完成")
    
    def convert_angles_from_robot(self, q_robot: List[float]) -> List[float]:
        """将实际机器人角度转换为求解器角度"""
        return [-angle for angle in q_robot]
    
    def set_joint_angles(self, joint_angles: np.ndarray):
        """设置关节角度"""
        self.mjdata.qpos[self.qpos_indices[:4]] = joint_angles
        mujoco.mj_forward(self.mjmodel, self.mjdata)
    
    def get_mujoco_position(self) -> np.ndarray:
        """获取MuJoCo末端位置"""
        return self.mjdata.xpos[self.end_effector_id].copy()
    
    def get_fk_position(self, joint_angles: np.ndarray) -> np.ndarray:
        """计算FK位置"""
        if not RTB_AVAILABLE or self.robot_model is None:
            return None
        
        # 转换为求解器角度
        solver_angles = self.convert_angles_from_robot(joint_angles.tolist())
        
        # 计算FK
        T = self.robot_model.fkine(solver_angles)
        fk_pos = T.t.tolist()
        
        # 应用Y轴坐标系转换
        corrected_pos = np.array([
            fk_pos[0],      # X轴保持不变
            -fk_pos[1],     # Y轴取反以匹配MuJoCo坐标系
            fk_pos[2]       # Z轴保持不变
        ])
        
        return corrected_pos
    
    def test_coordinate_offset(self):
        """测试坐标系偏移"""
        print("\n" + "="*60)
        print("🔬 坐标系偏移分析")
        print("="*60)
        
        # 测试用例：不同的关节角度配置
        test_cases = [
            ("Home位置", [0, -np.pi/2, np.pi/2, 0]),
            ("全零位置", [0, 0, 0, 0]),
            ("关节2=-45°", [0, -np.pi/4, 0, 0]),
            ("关节3=+45°", [0, 0, np.pi/4, 0]),
            ("关节1=+45°", [np.pi/4, -np.pi/2, np.pi/2, 0]),
            ("复合角度1", [np.pi/6, -np.pi/3, np.pi/3, np.pi/6]),
            ("复合角度2", [-np.pi/6, -2*np.pi/3, 2*np.pi/3, -np.pi/6]),
        ]
        
        offsets = []
        
        for i, (description, angles) in enumerate(test_cases):
            print(f"\n📍 测试案例 {i+1}: {description}")
            print(f"   关节角度: [{', '.join([f'{np.degrees(a):.1f}°' for a in angles])}]")
            
            # 设置关节角度
            self.set_joint_angles(np.array(angles))
            
            # 获取位置
            mujoco_pos = self.get_mujoco_position()
            fk_pos = self.get_fk_position(np.array(angles))
            
            if fk_pos is not None:
                # 计算偏移
                offset = mujoco_pos - fk_pos
                offsets.append(offset)
                
                print(f"   MuJoCo位置: X={mujoco_pos[0]*1000:.1f}mm, Y={mujoco_pos[1]*1000:.1f}mm, Z={mujoco_pos[2]*1000:.1f}mm")
                print(f"   FK计算位置: X={fk_pos[0]*1000:.1f}mm, Y={fk_pos[1]*1000:.1f}mm, Z={fk_pos[2]*1000:.1f}mm")
                print(f"   偏移量: ΔX={offset[0]*1000:+.1f}mm, ΔY={offset[1]*1000:+.1f}mm, ΔZ={offset[2]*1000:+.1f}mm")
                
                # 计算偏移幅度
                offset_magnitude = np.linalg.norm(offset)
                print(f"   偏移幅度: {offset_magnitude*1000:.1f}mm")
        
        # 分析偏移规律
        if offsets:
            self._analyze_offset_pattern(offsets)
    
    def _analyze_offset_pattern(self, offsets: List[np.ndarray]):
        """分析偏移模式"""
        print(f"\n" + "="*60)
        print("📊 偏移模式分析")
        print("="*60)
        
        offsets_array = np.array(offsets)
        
        # 计算统计信息
        mean_offset = np.mean(offsets_array, axis=0)
        std_offset = np.std(offsets_array, axis=0)
        max_offset = np.max(offsets_array, axis=0)
        min_offset = np.min(offsets_array, axis=0)
        
        print(f"📈 统计结果 (mm):")
        print(f"   平均偏移: ΔX={mean_offset[0]*1000:+.1f}±{std_offset[0]*1000:.1f}, ΔY={mean_offset[1]*1000:+.1f}±{std_offset[1]*1000:.1f}, ΔZ={mean_offset[2]*1000:+.1f}±{std_offset[2]*1000:.1f}")
        print(f"   最大偏移: ΔX={max_offset[0]*1000:+.1f}, ΔY={max_offset[1]*1000:+.1f}, ΔZ={max_offset[2]*1000:+.1f}")
        print(f"   最小偏移: ΔX={min_offset[0]*1000:+.1f}, ΔY={min_offset[1]*1000:+.1f}, ΔZ={min_offset[2]*1000:+.1f}")
        
        # 判断偏移类型
        mean_magnitude = np.linalg.norm(mean_offset)
        std_magnitude = np.linalg.norm(std_offset)
        
        print(f"\n🔍 偏移分析:")
        print(f"   平均偏移幅度: {mean_magnitude*1000:.1f}mm")
        print(f"   偏移变异度: {std_magnitude*1000:.1f}mm")
        
        if std_magnitude < 0.02:  # 20mm
            print(f"   ✅ 检测到固定偏移 - 可以通过常量校正解决")
            print(f"   💡 建议校正值: ΔX={-mean_offset[0]*1000:+.1f}mm, ΔY={-mean_offset[1]*1000:+.1f}mm, ΔZ={-mean_offset[2]*1000:+.1f}mm")
        else:
            print(f"   ⚠️  检测到变化偏移 - 可能需要复杂的坐标变换")
        
        return mean_offset
    
    def test_offset_correction(self, correction_offset: np.ndarray):
        """测试偏移校正效果"""
        print(f"\n" + "="*60)
        print("🔧 偏移校正测试")
        print("="*60)
        
        # 使用一个新的测试案例
        test_angles = [np.pi/8, -np.pi/3, np.pi/4, np.pi/12]
        print(f"测试角度: [{', '.join([f'{np.degrees(a):.1f}°' for a in test_angles])}]")
        
        # 设置角度并获取位置
        self.set_joint_angles(np.array(test_angles))
        mujoco_pos = self.get_mujoco_position()
        fk_pos = self.get_fk_position(np.array(test_angles))
        
        if fk_pos is not None:
            # 应用校正
            corrected_fk_pos = fk_pos + correction_offset
            
            # 计算误差
            original_error = np.linalg.norm(mujoco_pos - fk_pos)
            corrected_error = np.linalg.norm(mujoco_pos - corrected_fk_pos)
            
            print(f"MuJoCo位置: X={mujoco_pos[0]*1000:.1f}mm, Y={mujoco_pos[1]*1000:.1f}mm, Z={mujoco_pos[2]*1000:.1f}mm")
            print(f"原始FK位置: X={fk_pos[0]*1000:.1f}mm, Y={fk_pos[1]*1000:.1f}mm, Z={fk_pos[2]*1000:.1f}mm")
            print(f"校正FK位置: X={corrected_fk_pos[0]*1000:.1f}mm, Y={corrected_fk_pos[1]*1000:.1f}mm, Z={corrected_fk_pos[2]*1000:.1f}mm")
            print(f"原始误差: {original_error*1000:.1f}mm")
            print(f"校正误差: {corrected_error*1000:.1f}mm")
            print(f"改善程度: {((original_error-corrected_error)/original_error*100):.1f}%")

def main():
    """主程序"""
    print("🔬 机械臂坐标系偏移调试工具")
    print("="*60)
    
    xml_path = "scene.xml"
    if not os.path.exists(xml_path):
        print(f"❌ 找不到XML文件: {xml_path}")
        return
    
    if not RTB_AVAILABLE:
        print("❌ 请安装 roboticstoolbox-python 和 spatialmath-python")
        return
    
    try:
        # 创建调试器
        debugger = CoordinateDebugger(xml_path)
        
        # 执行坐标系偏移测试
        debugger.test_coordinate_offset()
        
        print(f"\n" + "="*60)
        print("🎯 调试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 调试过程错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 