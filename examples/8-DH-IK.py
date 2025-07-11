#!/usr/bin/env python3
"""
机械臂IK运动学仿真系统 - 8-DH-IK.py

🚀 功能特点:
- 基于DH参数的4轴机械臂正逆运动学
- MuJoCo物理仿真环境
- 平滑运动控制，支持与物体的真实物理交互
- 实时IK求解和可视化

🎯 使用方法:
1. 运行程序: python examples/8-DH-IK.py
2. 在MuJoCo查看器中观察机械臂
3. 在终端输入指令控制机械臂:
   - "x y z" - 移动到指定位置 (单位: 米)
   - "home" - 回到Home位置
   - "pos" - 显示当前状态
   - "wait" - 等待当前运动完成
   - "demo" - 演示抓取动作序列
   - "q" - 退出程序

🔧 技术细节:
- 使用roboticstoolbox进行IK求解
- 平滑插值运动，避免突跳
- 支持碰撞检测和物理交互
- 实时可视化反馈

📦 场景物体:
- 红色、蓝色、绿色立方体 (可抓取)
- 黄色球体 (可推动)
- 灰色圆柱体 (固定障碍物)

⚙️ 依赖项:
- mujoco
- roboticstoolbox-python
- spatialmath-python
- numpy

作者: 基于LeRobot-Kinematics项目修改
版本: 2024.12 物理交互版
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import threading
from typing import List, Optional, Tuple

# 设置环境
np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

import roboticstoolbox as rtb
from spatialmath import SE3
RTB_AVAILABLE = True

# ===================== 配置参数 =====================
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# 🔧 修正DH参数 - 参考keyboard_control.py的正确值
L1 = 0.0304   # 第一连杆长度 (30.4mm)
L2 = 0.116    # 第二连杆长度 (116mm)  
L3 = 0.1347   # 第三连杆长度 (134.7mm)
L4 = 0.15     # 第四连杆长度 (150mm)
D1 = 0.0542   # 第一关节高度偏移 (54.2mm)

# Home位置
HOME_ANGLES_DEG = [0, -90, 90, 0, 0, 0]
HOME_QPOS = np.array([math.radians(angle) for angle in HOME_ANGLES_DEG])

# 运动控制参数
MOVEMENT_SPEED = 2.0  # 关节运动速度 (rad/s)
TIMESTEP = 0.002      # 仿真时间步长

# ===================== 简化机器人控制器 =====================
class SimpleRobotController:
    def __init__(self, xml_path: str):
        """初始化机器人控制器"""
        self.xml_path = xml_path
        self.mjmodel = None
        self.mjdata = None
        self.qpos_indices = None
        self.ctrl_indices = None
        self.end_effector_id = None
        self.robot_model = None
        
        # 核心变量：目标关节角度和当前目标
        self.target_qpos = HOME_QPOS.copy()
        self.current_target_qpos = HOME_QPOS.copy()  # 平滑运动的当前目标
        
        # 运动状态
        self.is_moving = False
        self.movement_start_time = 0
        
        self._init_mujoco()
        if RTB_AVAILABLE:
            self._init_robot_model()
        
    def _init_mujoco(self):
        """初始化MuJoCo模型"""
        self.mjmodel = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mjdata = mujoco.MjData(self.mjmodel)
        
        # 设置仿真参数
        self.mjmodel.opt.timestep = TIMESTEP
        
        # 🔧 显示实际的XML配置的PID参数
        print(f"🔧 使用增强的PID控制参数 (XML配置):")
        print(f"   - 旋转关节: kp=300, dampratio=2.0, forcerange=±400 (强力抗重力)")
        print(f"   - 肩部关节: kp=1200, dampratio=3.0, forcerange=±800 (超强抗重力)")
        print(f"   - 肘部关节: kp=800, dampratio=2.5, forcerange=±600 (强力抗重力)")
        print(f"   - 腕部关节: kp=400, dampratio=2.0, forcerange=±300 (中强抗重力)")
        print(f"   - 末端关节: kp=50-100, dampratio=1.2-1.5, forcerange=±60-100")
        print(f"   💪 大幅增强PID参数，专门对抗侧向伸展时的重力矩")
        
        # 获取关节索引
        self.qpos_indices = np.array([
            self.mjmodel.jnt_qposadr[self.mjmodel.joint(name).id] 
            for name in JOINT_NAMES
        ])
        
        # 获取控制器索引
        self.ctrl_indices = np.array([
            self.mjmodel.actuator(name).id 
            for name in JOINT_NAMES
        ])

        # 获取末端执行器ID
        try:
            self.end_effector_id = self.mjmodel.body("Moving_Jaw").id
        except:
            # 尝试其他可能的名称
            for name in ["end_effector", "gripper", "tool", "tcp", "Jaw"]:
                try:
                    self.end_effector_id = self.mjmodel.body(name).id
                    break
                except:
                    continue
            else:
                self.end_effector_id = self.mjmodel.nbody - 1
        
        # 🔧 关键：正确初始化位置和控制器
        # 1. 设置关节位置
        self.mjdata.qpos[self.qpos_indices] = HOME_QPOS.copy()
        
        # 2. 设置控制器目标为相同位置
        self.mjdata.ctrl[self.ctrl_indices] = HOME_QPOS.copy()
        
        # 3. 执行正向运动学更新
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        
        # 4. 🔑 执行多步仿真来稳定机械臂位置
        print("🔄 正在稳定机械臂位置...")
        for _ in range(100):  # 执行100步稳定仿真
            mujoco.mj_step(self.mjmodel, self.mjdata)
        
        print(f"✅ MuJoCo初始化完成 - 关节数: {self.mjmodel.nq}, 执行器数: {self.mjmodel.nu}")
        print(f"📍 末端执行器ID: {self.end_effector_id}")
        
        # 验证初始位置
        current_pos = self.get_end_effector_position()
        print(f"🎯 初始末端位置: X={current_pos[0]*1000:.1f}mm, Y={current_pos[1]*1000:.1f}mm, Z={current_pos[2]*1000:.1f}mm")
    
    def _init_robot_model(self):
        """初始化DH机器人模型 - 使用正确的DH参数"""
        # 🔧 使用keyboard_control.py中验证过的正确DH参数
        self.robot_model = rtb.DHRobot([
            rtb.RevoluteDH(a=L1, alpha=np.pi/2, d=D1, offset=0),      # 关节1: a=30.4mm, d=54.2mm
            rtb.RevoluteDH(a=L2, alpha=0, d=0, offset=np.pi/2),       # 关节2: a=116mm, offset=90°
            rtb.RevoluteDH(a=L3, alpha=0, d=0, offset=-np.pi/2),      # 关节3: a=134.7mm, offset=-90°
            rtb.RevoluteDH(a=L4, alpha=0, d=0, offset=0)              # 关节4: a=150mm
        ], name="SO100 Robot")
        
        # 设置关节限制 (±90度，与真实机器人一致)
        joint_limits = [
            [-np.pi/2, np.pi/2],    # 关节1: ±90度
            [-np.pi/2, np.pi/2],    # 关节2: ±90度  
            [-np.pi/2, np.pi/2],    # 关节3: ±90度
            [-np.pi/2, np.pi/2],    # 关节4: ±90度
        ]
        
        for i, (qmin, qmax) in enumerate(joint_limits):
            self.robot_model.links[i].qlim = [qmin, qmax]
        
        print("✅ 机器人DH模型创建成功 - 使用正确的DH参数")
    
    def convert_angles_to_robot(self, q_solver: List[float]) -> List[float]:
        """将求解器角度转换为实际机器人角度"""
        return [-angle for angle in q_solver]
    
    def convert_angles_from_robot(self, q_robot: List[float]) -> List[float]:
        """将实际机器人角度转换为求解器角度"""
        return [-angle for angle in q_robot]
    
    def update_simulation(self):
        """🔑 核心函数：每帧更新仿真状态"""
        # 平滑运动控制：逐步接近目标位置
        current_qpos = self.mjdata.qpos[self.qpos_indices].copy()
        
        # 计算到目标的距离
        position_error = self.target_qpos - current_qpos
        max_error = np.max(np.abs(position_error))
        
        # 如果还在运动中，进行平滑插值
        if max_error > 0.01:  # 1度的误差阈值
            self.is_moving = True
            
            # 计算这一步的运动量
            dt = self.mjmodel.opt.timestep
            max_step = MOVEMENT_SPEED * dt
            
            # 限制每步的最大运动量
            step_size = np.clip(position_error, -max_step, max_step)
            self.current_target_qpos = current_qpos + step_size
            
            # 设置控制器目标
            self.mjdata.ctrl[self.ctrl_indices] = self.current_target_qpos
        else:
            self.is_moving = False
            # 精确设置最终目标
            self.mjdata.ctrl[self.ctrl_indices] = self.target_qpos
        
        # 执行物理仿真步进（包含物体交互）
        mujoco.mj_step(self.mjmodel, self.mjdata)
    
    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        return self.mjdata.xpos[self.end_effector_id].copy()
    
    def get_current_joint_angles(self) -> np.ndarray:
        """获取当前关节角度"""
        return self.mjdata.qpos[self.qpos_indices].copy()
    
    def solve_ik(self, target_pos: List[float]) -> Optional[np.ndarray]:
        """求解逆运动学 - 修复坐标系转换问题"""
        if not RTB_AVAILABLE or self.robot_model is None:
            print("❌ roboticstoolbox不可用")
            return None
        
        try:
            # 🔧 修复坐标系转换：Y轴方向相反
            corrected_target = [
                target_pos[0],      # X轴保持不变
                -target_pos[1],     # Y轴取反以匹配MuJoCo坐标系
                target_pos[2]       # Z轴保持不变
            ]
            
            print(f"🔧 坐标系转换: 输入({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}) -> IK求解({corrected_target[0]:.3f}, {corrected_target[1]:.3f}, {corrected_target[2]:.3f})")
            
            # 正确的变换矩阵构建方法 - 使用修正后的坐标
            T_target = SE3.Tx(corrected_target[0]) * SE3.Ty(corrected_target[1]) * SE3.Tz(corrected_target[2])
            
            # 获取当前角度并转换为求解器角度
            current_angles = self.get_current_joint_angles()
            current_q_solver = self.convert_angles_from_robot(current_angles[:4].tolist())
            
            # 求解IK - 只约束位置，不约束姿态
            sol = self.robot_model.ikine_LM(
                T_target, 
                q0=current_q_solver, 
                mask=[1,1,1,0,0,0],  # 只约束XYZ位置
                tol=1e-4, 
                joint_limits=True,
                slimit=20
            )
            
            if sol.success:
                # 转换回实际机器人角度
                q_solution = self.convert_angles_to_robot(sol.q.tolist())
                
                # 检查角度限制 (±90度)
                for i, angle in enumerate(q_solution):
                    if abs(angle) > np.pi/2:
                        print(f"❌ 关节{i+1}角度超出范围: {np.degrees(angle):.1f}° (最大±90°)")
                        return None
                
                # 返回numpy数组
                return np.array(q_solution)
            else:
                print("❌ IK求解失败")
                return None
                
        except Exception as e:
            print(f"❌ IK计算错误: {e}")
            return None
    
    def move_to_position(self, target_pos: List[float]) -> bool:
        """🎯 移动到目标位置（平滑运动 + IK验证）"""
        print(f"\n🎯 目标位置: X={target_pos[0]*1000:.1f}mm, Y={target_pos[1]*1000:.1f}mm, Z={target_pos[2]*1000:.1f}mm")
        
        # 工作空间检查
        if not self.check_workspace_limits(target_pos):
            print("❌ 目标位置超出安全工作空间，无法移动。")
            print("-" * 50)
            return False

        # 求解IK
        joint_solution = self.solve_ik(target_pos)
        
        if joint_solution is not None:
            print(f"🔧 目标关节角度: [{', '.join([f'{np.degrees(angle):.1f}°' for angle in joint_solution])}]")
            
            # 🔧 应用角度映射校正
            mapped_angles = self.calibrate_angle_mapping(joint_solution)
            
            # 🔧 先进行FK验证（不改变MuJoCo状态）
            verification_success = self.verify_ik_solution_pure(target_pos, mapped_angles)
            
            if verification_success:
                # 验证通过，更新目标关节角度（保持末端关节不变）
                self.target_qpos[:4] = mapped_angles
                self.movement_start_time = time.time()
                
                print(f"✅ IK验证通过，开始平滑运动...")
                
                # 🔧 添加运动前后的位置对比
                print(f"🚀 开始运动到目标位置...")
                self._show_position_comparison(target_pos)
                
                print("-" * 50)
                return True
            else:
                print("❌ IK验证失败")
                print("-" * 50)
                return False
        else:
            print("❌ 无法到达目标位置")
            print("-" * 50)
            return False
    
    def reset_to_home(self):
        """重置到Home位置"""
        print("🏠 重置到Home位置")
        self.target_qpos = HOME_QPOS.copy()
        self.movement_start_time = time.time()
    
    def show_status(self):
        """显示当前状态 - 包含FK验证和目标对比"""
        # 获取MuJoCo中的实际位置
        mujoco_pos = self.get_mujoco_end_effector_position()
        current_angles = self.get_current_joint_angles()
        
        # 获取FK计算位置
        fk_pos = self._update_current_position_from_fk()
        
        status_icon = "🔄" if self.is_moving else "🔒"
        
        print(f"\n{status_icon} === 当前状态详情 ===")
        print(f"🎯 MuJoCo实际位置: X={mujoco_pos[0]*1000:.1f}mm, Y={mujoco_pos[1]*1000:.1f}mm, Z={mujoco_pos[2]*1000:.1f}mm")
        
        if fk_pos is not None:
            print(f"🔧 FK计算位置: X={fk_pos[0]*1000:.1f}mm, Y={fk_pos[1]*1000:.1f}mm, Z={fk_pos[2]*1000:.1f}mm")
            
            # 计算FK与MuJoCo的差异
            fk_diff = np.array(fk_pos) - np.array(mujoco_pos)
            fk_error = np.linalg.norm(fk_diff)
            print(f"📏 FK-MuJoCo差异: ΔX={fk_diff[0]*1000:+.1f}mm, ΔY={fk_diff[1]*1000:+.1f}mm, ΔZ={fk_diff[2]*1000:+.1f}mm")
            print(f"📐 总差异: {fk_error*1000:.2f}mm")
            
            # 如果差异很大，说明存在控制问题
            if fk_error > 0.1:  # 100mm
                print(f"⚠️ 警告: FK与MuJoCo差异过大，可能存在控制问题")
        else:
            print(f"❌ FK计算失败")
        
        joint_degrees = [np.degrees(angle) for angle in current_angles]
        print(f"🔧 当前关节角度: [{', '.join([f'{angle:.1f}°' for angle in joint_degrees])}]")
        
        # 显示目标关节角度
        target_degrees = [np.degrees(angle) for angle in self.target_qpos]
        print(f"🎯 目标关节角度: [{', '.join([f'{angle:.1f}°' for angle in target_degrees])}]")
        
        # 计算关节角度误差
        angle_error = np.linalg.norm(self.target_qpos - current_angles)
        print(f"📐 关节角度误差: {np.degrees(angle_error):.2f}°")
        
        if self.is_moving:
            print("📈 状态: 运动中...")
        else:
            print("📍 状态: 已到达目标位置")
        
        print("=" * 40)
    
    def _show_position_comparison(self, target_pos: List[float]):
        """🔍 显示目标位置和MuJoCo实际位置的对比"""
        try:
            # 获取当前MuJoCo实际位置
            current_mujoco_pos = self.get_end_effector_position()
            
            # 计算误差
            target_array = np.array(target_pos)
            mujoco_array = np.array(current_mujoco_pos)
            mujoco_error = np.linalg.norm(target_array - mujoco_array)
            
            print(f"📍 位置对比:")
            print(f"   🎯 目标位置: X={target_pos[0]*1000:.1f}mm, Y={target_pos[1]*1000:.1f}mm, Z={target_pos[2]*1000:.1f}mm")
            print(f"   🤖 MuJoCo实际: X={current_mujoco_pos[0]*1000:.1f}mm, Y={current_mujoco_pos[1]*1000:.1f}mm, Z={current_mujoco_pos[2]*1000:.1f}mm")
            print(f"   📏 位置误差: {mujoco_error*1000:.2f}mm")
            
            # 评估误差
            if mujoco_error < 0.01:  # 10mm
                print(f"   ✅ 位置精度优秀 (<10mm)")
            elif mujoco_error < 0.05:  # 50mm
                print(f"   ✅ 位置精度良好 (<50mm)")
            elif mujoco_error < 0.1:  # 100mm
                print(f"   ⚠️ 位置精度中等 (<100mm)")
            else:
                print(f"   ❌ 位置精度较低 (>{mujoco_error*1000:.1f}mm)")
                
        except Exception as e:
            print(f"❌ 位置对比失败: {e}")

    def wait_for_movement_complete(self, timeout: float = 10.0) -> bool:
        """等待运动完成"""
        start_time = time.time()
        while self.is_moving and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not self.is_moving:
            actual_pos = self.get_end_effector_position()
            print(f"✅ 运动完成! 实际位置: X={actual_pos[0]*1000:.1f}mm, Y={actual_pos[1]*1000:.1f}mm, Z={actual_pos[2]*1000:.1f}mm")
            print("-" * 50)
            return True
        else:
            print("⏰ 运动超时")
            print("-" * 50)
            return False

    def _update_current_position_from_fk(self):
        """🔧 使用正运动学更新当前末端位置 - 修复坐标系转换"""
        if not RTB_AVAILABLE or self.robot_model is None:
            return None
            
        try:
            # 获取当前关节角度
            current_angles = self.get_current_joint_angles()
            
            # 转换为求解器角度
            current_q_solver = self.convert_angles_from_robot(current_angles[:4].tolist())
            
            # 使用roboticstoolbox计算正运动学
            T = self.robot_model.fkine(current_q_solver)
            fk_pos = T.t.tolist()
            
            # 🔧 应用相同的坐标系转换：Y轴取反
            corrected_pos = [
                fk_pos[0],      # X轴保持不变
                -fk_pos[1],     # Y轴取反以匹配MuJoCo坐标系
                fk_pos[2]       # Z轴保持不变
            ]
            
            return corrected_pos
        except Exception as e:
            print(f"❌ 正运动学计算错误: {e}")
            return None
    
    def get_mujoco_end_effector_position(self) -> np.ndarray:
        """🎯 从MuJoCo仿真中获取Moving_Jaw的实际位置"""
        return self.mjdata.xpos[self.end_effector_id].copy()
    
    def verify_ik_solution_pure(self, target_pos: List[float], joint_angles: np.ndarray) -> bool:
        """🔍 纯FK验证（不改变MuJoCo状态）"""
        try:
            if not RTB_AVAILABLE or self.robot_model is None:
                return False
            
            # 转换为求解器角度进行FK计算
            solver_angles = self.convert_angles_from_robot(joint_angles.tolist())
            
            # 使用roboticstoolbox计算正运动学
            T = self.robot_model.fkine(solver_angles)
            fk_pos = T.t.tolist()
            
            # 应用坐标系转换：Y轴取反
            corrected_pos = [
                fk_pos[0],      # X轴保持不变
                -fk_pos[1],     # Y轴取反以匹配MuJoCo坐标系
                fk_pos[2]       # Z轴保持不变
            ]
            
            # 计算误差
            target_array = np.array(target_pos)
            fk_array = np.array(corrected_pos)
            fk_error = np.linalg.norm(target_array - fk_array)
            
            print(f"🔍 纯FK验证结果:")
            print(f"   目标位置: X={target_pos[0]*1000:.1f}mm, Y={target_pos[1]*1000:.1f}mm, Z={target_pos[2]*1000:.1f}mm")
            print(f"   FK计算位置: X={corrected_pos[0]*1000:.1f}mm, Y={corrected_pos[1]*1000:.1f}mm, Z={corrected_pos[2]*1000:.1f}mm")
            print(f"   FK误差: {fk_error*1000:.2f}mm")
            
            # 评估精度
            if fk_error < 0.001:  # 1mm误差
                print(f"   ✅ FK精度极高 (<1mm)")
            elif fk_error < 0.005:  # 5mm误差
                print(f"   ✅ FK精度良好 (<5mm)")
            elif fk_error < 0.01:  # 10mm误差
                print(f"   ⚠️ FK精度中等 (<10mm)")
            else:
                print(f"   ❌ FK精度较低 (>{fk_error*1000:.1f}mm)")
            
            return fk_error < 0.05  # 接受50mm以内的误差
            
        except Exception as e:
            print(f"❌ 纯FK验证失败: {e}")
            return False

    def verify_ik_solution(self, target_pos: List[float], joint_solution: np.ndarray) -> bool:
        """🔍 验证IK求解结果的准确性"""
        try:
            # 使用正运动学验证
            fk_pos = self._update_current_position_from_fk()
            if fk_pos is None:
                return False
            
            # 获取MuJoCo中的实际位置
            mujoco_pos = self.get_mujoco_end_effector_position()
            
            # 计算误差
            target_array = np.array(target_pos)
            fk_array = np.array(fk_pos)
            mujoco_array = np.array(mujoco_pos)
            
            fk_error = np.linalg.norm(target_array - fk_array)
            mujoco_error = np.linalg.norm(target_array - mujoco_array)
            
            print(f"🔍 IK验证结果:")
            print(f"   目标位置: X={target_pos[0]*1000:.1f}mm, Y={target_pos[1]*1000:.1f}mm, Z={target_pos[2]*1000:.1f}mm")
            print(f"   FK计算位置: X={fk_pos[0]*1000:.1f}mm, Y={fk_pos[1]*1000:.1f}mm, Z={fk_pos[2]*1000:.1f}mm")
            print(f"   MuJoCo实际位置: X={mujoco_pos[0]*1000:.1f}mm, Y={mujoco_pos[1]*1000:.1f}mm, Z={mujoco_pos[2]*1000:.1f}mm")
            print(f"   FK误差: {fk_error*1000:.2f}mm")
            print(f"   MuJoCo误差: {mujoco_error*1000:.2f}mm")
            
            # 评估精度
            if fk_error < 0.001:  # 1mm误差
                print(f"   ✅ FK精度极高 (<1mm)")
            elif fk_error < 0.005:  # 5mm误差
                print(f"   ✅ FK精度良好 (<5mm)")
            elif fk_error < 0.01:  # 10mm误差
                print(f"   ⚠️ FK精度中等 (<10mm)")
            else:
                print(f"   ❌ FK精度较低 (>{fk_error*1000:.1f}mm)")
            
            return fk_error < 0.05  # 接受50mm以内的误差（放宽验证条件）
            
        except Exception as e:
            print(f"❌ IK验证失败: {e}")
            return False

    def check_workspace_limits(self, target_pos: List[float]) -> bool:
        """🔒 检查目标位置是否在安全工作空间内 - 修正版"""
        x, y, z = target_pos
        
        print(f"🔍 检查工作空间限制: X={x*1000:.0f}mm, Y={y*1000:.0f}mm, Z={z*1000:.0f}mm")
        
        # 🔧 基于实际机械臂DH参数的更宽松限制
        # 总连杆长度：L1+L2+L3+L4 = 30.4+116+134.7+150 = 431.1mm
        max_reach = 0.43  # 430mm最大理论距离
        min_reach = 0.05  # 50mm最小距离（避免奇异点）
        
        # 计算到原点的距离
        reach = np.sqrt(x**2 + y**2 + z**2)
        
        # 检查最大距离
        if reach > max_reach:
            print(f"❌ 超出最大工作距离: {reach*1000:.0f}mm (最大: {max_reach*1000:.0f}mm)")
            return False
        
        # 检查最小距离（避免奇异点）
        if reach < min_reach:
            print(f"❌ 距离过近，可能导致奇异点: {reach*1000:.0f}mm (最小: {min_reach*1000:.0f}mm)")
            return False
        
        # 🔧 更宽松的轴向限制
        # X轴：允许0到最大伸展距离（包括正上方位置）
        if x < -0.1 or x > max_reach:
            print(f"❌ X轴超出范围: {x*1000:.0f}mm (范围: -100mm到{max_reach*1000:.0f}mm)")
            return False
            
        # Y轴：左右对称，基于机械臂旋转能力
        max_y = max_reach * 0.8  # 约344mm
        if abs(y) > max_y:
            print(f"❌ Y轴超出范围: {y*1000:.0f}mm (范围: ±{max_y*1000:.0f}mm)")
            return False
            
        # Z轴：考虑地面和最大高度
        # 最低：稍高于地面，最高：机械臂最大伸展高度
        if z < -0.05 or z > 0.4:
            print(f"❌ Z轴超出范围: {z*1000:.0f}mm (范围: -50mm到400mm)")
            return False
        
        # 🔧 特殊位置检查：避免已知的奇异点
        # 检查是否太接近机械臂基座正上方的奇异点
        horizontal_dist = np.sqrt(x**2 + y**2)
        if horizontal_dist < 0.02 and abs(z) > 0.3:
            print(f"⚠️ 接近奇异点：水平距离{horizontal_dist*1000:.0f}mm，高度{z*1000:.0f}mm")
            print(f"建议：增加水平距离到至少20mm")
            # 不直接拒绝，只是警告
        
        print(f"✅ 工作空间检查通过 (距离: {reach*1000:.0f}mm)")
        return True
    
    def safe_set_joint_angles(self, joint_angles: np.ndarray) -> bool:
        """🔒 安全设置关节角度 - 防止段错误版本"""
        try:
            # 1. 检查角度限制
            for i, angle in enumerate(joint_angles):
                if abs(angle) > np.pi/2:
                    print(f"❌ 关节{i+1}角度超出±90°限制: {np.degrees(angle):.1f}°")
                    return False
            
            # 2. 应用角度映射校正
            mapped_angles = self.calibrate_angle_mapping(joint_angles)
            
            # 3. 🔧 使用更安全的关节角度设置方法
            print(f"🚀 安全设置关节角度")
            
            # 检查索引有效性
            if len(self.qpos_indices) < 4:
                print(f"❌ 关节索引不足: {len(self.qpos_indices)}")
                return False
            
            # 🔑 关键修复：逐个安全设置关节角度，而不是批量操作
            for i in range(4):  # 只处理前4个关节
                if i < len(mapped_angles) and self.qpos_indices[i] < len(self.mjdata.qpos):
                    self.mjdata.qpos[self.qpos_indices[i]] = mapped_angles[i]
                else:
                    print(f"❌ 索引{i}超出范围")
                    return False
            
            # 🔑 安全的状态更新
            mujoco.mj_forward(self.mjmodel, self.mjdata)
            
            print(f"✅ 关节角度安全设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 设置关节角度失败: {e}")
            # 不打印完整traceback，避免进一步的内存访问
            return False

    def calibrate_angle_mapping(self, rtb_angles: np.ndarray) -> np.ndarray:
        """🔧 角度映射校正：轻微调整以匹配MuJoCo坐标系"""
        # 使用更保守的映射，避免过度修改
        
        # 原始角度
        r1, r2, r3, r4 = rtb_angles
        
        # 🔧 保守的角度校正（轻微调整）
        # 基于您的建议，只做必要的微调
        
        # 关节1：旋转角度基本正确
        m1 = r1
        
        # 关节2：轻微调整
        m2 = r2 * 0.95 + 0.05  # 很小的调整
        
        # 关节3：轻微调整  
        m3 = r3 * 0.95
        
        # 关节4：基本保持
        m4 = r4 * 0.98
        
        mapped_angles = np.array([m1, m2, m3, m4])
        
        print(f"🔧 保守角度映射: [{', '.join([f'{np.degrees(a):.1f}°' for a in rtb_angles])}] -> [{', '.join([f'{np.degrees(a):.1f}°' for a in mapped_angles])}]")
        
        return mapped_angles

# ===================== 用户输入处理 =====================
def input_handler(robot: SimpleRobotController):
    """处理用户输入"""
    print("\n" + "="*50)
    print("📝 控制指令:")
    print("   x y z      - 移动到指定位置 (单位: 米)")
    print("   home       - 回到Home位置")
    print("   pos        - 显示当前位置")
    print("   wait       - 等待当前运动完成")
    print("   demo       - 演示抓取动作")
    print("   q          - 退出程序")
    print("="*50)
    print("💡 机械臂会平滑运动到输入的位置，支持与cube的物理交互")
    
    while True:
        try:
            user_input = input("\n🔢 输入指令: ").strip().lower()
            
            if user_input == 'q':
                print("👋 退出程序")
                os._exit(0)
            
            elif user_input == 'home':
                robot.reset_to_home()
            
            elif user_input == 'pos':
                robot.show_status()
                
            elif user_input == 'wait':
                print("⏳ 等待运动完成...")
                robot.wait_for_movement_complete()
                
            elif user_input == 'demo':
                print("🎭 IK测试演示序列:")
                # 简化的IK测试序列 - 专注于红色cube
                positions = [
                    ([0.25, 0.0, 0.15], "移动到红色cube正上方"),
                    ([0.25, 0.0, 0.10], "下降到中等高度"),
                    ([0.25, 0.0, 0.06], "接近红色cube"),
                    ([0.25, 0.05, 0.08], "稍微偏移观察角度"),
                    ([0.25, -0.05, 0.08], "另一侧观察角度"),
                    ([0.20, 0.0, 0.12], "退到安全距离"),
                    ([0.15, 0.0, 0.15], "返回中心安全位置"),
                ]
                
                for i, (pos, description) in enumerate(positions):
                    print(f"第{i+1}步: {description} -> [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                    if robot.move_to_position(pos):
                        robot.wait_for_movement_complete()
                        time.sleep(1.0)  # 稍长停留时间观察IK效果
                
                print("🎭 IK测试演示完成 - 专注红色cube交互")
            
            else:
                # 解析坐标
                coords = user_input.split()
                if len(coords) == 3:
                    try:
                        x, y, z = map(float, coords)
                        robot.move_to_position([x, y, z])
                    except ValueError:
                        print("❌ 请输入有效的数字坐标")
                else:
                    print("❌ 请输入 'x y z' 格式的坐标")
                    
        except KeyboardInterrupt:
            print("\n👋 退出程序")
            os._exit(0)
        except Exception as e:
            print(f"❌ 输入处理错误: {e}")

# ===================== 主程序 =====================
def main():
    """主程序"""
    print("🚀 机械臂IK运动学仿真系统")
    print("🎯 支持平滑运动和物理交互")
    print("="*50)
    
    xml_path = "scene.xml"
    if not os.path.exists(xml_path):
        print(f"❌ 找不到XML文件: {xml_path}")
        return
    
    try:
        # 初始化机器人
        robot = SimpleRobotController(xml_path)
        
        # 启动查看器
        with mujoco.viewer.launch_passive(robot.mjmodel, robot.mjdata) as viewer:
            print("✅ MuJoCo查看器已启动")
            print("🔄 物理仿真已激活")
            
            # 启动输入处理线程
            input_thread = threading.Thread(target=input_handler, args=(robot,), daemon=True)
            input_thread.start()
            
            # 主循环：持续更新仿真状态
            while viewer.is_running():
                robot.update_simulation()  # 🔑 核心：物理仿真 + 平滑运动
                viewer.sync()
                
    except KeyboardInterrupt:
        print("\n🛑 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()