#!/usr/bin/env python3
"""
SO101四轴机器人键盘控制程序
通过键盘控制末端执行器位置，使用逆运动学求解并发送到实际机器人
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

# 导入机器人控制
try:
    from lerobot.common.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    ROBOT_AVAILABLE = True
except ImportError:
    print("警告: 无法导入机器人控制模块，将只在仿真模式下运行")
    ROBOT_AVAILABLE = False

# 导入逆运动学
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
    IK_AVAILABLE = True
except ImportError:
    print("错误: 请安装 roboticstoolbox-python")
    exit(1)

# 导入键盘输入处理
KEYBOARD_AVAILABLE = False
PYNPUT_AVAILABLE = False

# 优先尝试pynput（无需root权限）
try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
    print("✅ 成功导入pynput库（无需root权限）")
except ImportError:
    print("警告: 无法导入pynput库，可以安装: pip install pynput")

# keyboard库作为备选（需要root权限）
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
    print("✅ 成功导入keyboard库（需要root权限）")
except ImportError:
    print("警告: 无法导入keyboard库")

if not KEYBOARD_AVAILABLE and not PYNPUT_AVAILABLE:
    print("将使用基本输入模式")


@dataclass
class RobotState:
    """机器人状态"""
    current_q: List[float]  # 当前关节角度 (实际机器人角度)
    current_pos: List[float]  # 当前末端位置 [x, y, z]
    target_pos: List[float]  # 目标末端位置 [x, y, z]
    is_moving: bool = False


class KeyboardController:
    """键盘控制器"""
    
    def __init__(self, robot_port: str = "/dev/ttyACM0", step_size: float = 0.005):
        self.step_size = step_size  # 每次移动的步长（米）
        self.robot_port = robot_port
        
        # 初始化机器人模型（用于逆运动学）
        self.robot_model = self._create_robot_model()
        
        # 机器人状态（所有关节0度时的位置）
        self.state = RobotState(
            current_q=[0.0, 0.0, 0.0, 0.0],  # 所有关节0度
            current_pos=[0.0, 0.0, 0.0],     # 初始位置，稍后会根据实际机器人状态更新
            target_pos=[0.0, 0.0, 0.0]
        )
        
        # 控制标志
        self.running = True
        self.auto_send = True  # 是否自动发送到实际机器人
        
        # 初始化实际机器人（如果可用）
        self.real_robot: Optional[SO101Follower] = None
        if ROBOT_AVAILABLE:
            self._init_real_robot()
        
        # 更新初始位置（如果没有连接真实机器人，也需要计算0度位置）
        if self.real_robot is None:
            self._update_current_position()
            self.state.target_pos = self.state.current_pos.copy()
            print(f"📍 仿真模式初始位置: X={self.state.current_pos[0]*1000:.1f}mm, Y={self.state.current_pos[1]*1000:.1f}mm, Z={self.state.current_pos[2]*1000:.1f}mm")
    
    def _create_robot_model(self):
        """创建机器人模型"""
        robot = rtb.DHRobot([
            rtb.RevoluteDH(a=0.0304, alpha=np.pi/2, d=0.0542, offset=0),  # 关节1
            rtb.RevoluteDH(a=0.116, alpha=0, d=0, offset=np.pi/2),        # 关节2
            rtb.RevoluteDH(a=0.1347, alpha=0, d=0, offset=-np.pi/2),      # 关节3
            rtb.RevoluteDH(a=0.15, alpha=0, d=0, offset=0)               # 关节4
        ], name="SO101 Robot")
        
        # 设置关节限制
        joint_limits = [
            [-np.pi/2, np.pi/2],    # 关节1: ±90度
            [-np.pi/2, np.pi/2],    # 关节2: ±90度  
            [-np.pi/2, np.pi/2],    # 关节3: ±90度
            [-np.pi/2, np.pi/2],    # 关节4: ±90度
        ]
        
        for i, (qmin, qmax) in enumerate(joint_limits):
            robot.links[i].qlim = [qmin, qmax]
        
        return robot
    
    def _init_real_robot(self):
        """初始化实际机器人"""
        try:
            robot_config = SO101FollowerConfig(
                port=self.robot_port,
                id="follower_arm_o"
            )
            self.real_robot = SO101Follower(robot_config)
            self.real_robot.connect()
            print(f"✅ 实际机器人连接成功: {self.robot_port}")
            
            # 将机器人移动到安全的初始位置
            self._move_to_initial_position()
            
        except Exception as e:
            print(f"❌ 实际机器人连接失败: {e}")
            self.real_robot = None
    
    def _move_to_initial_position(self):
        """将真实机器人移动到安全的初始位置"""
        if self.real_robot is None:
            return
            
        try:
            print("🔄 正在将机器人移动到初始位置...")
            
            # 初始角度：所有关节都在0度（安全位置）
            initial_angles_deg = [0.0, 0.0, 0.0, 0.0]
            
            # 发送到机器人
            action = {
                "shoulder_pan.pos": initial_angles_deg[0],
                "shoulder_lift.pos": initial_angles_deg[1], 
                "elbow_flex.pos": initial_angles_deg[2],
                "wrist_flex.pos": initial_angles_deg[3],
            }
            
            self.real_robot.send_action(action)
            print(f"✅ 机器人已移动到初始位置: {initial_angles_deg}")
            
            # 更新软件状态以匹配真实机器人
            self.state.current_q = [np.radians(angle) for angle in initial_angles_deg]
            self._update_current_position()
            self.state.target_pos = self.state.current_pos.copy()
            
            print(f"📍 初始末端位置: X={self.state.current_pos[0]*1000:.1f}mm, Y={self.state.current_pos[1]*1000:.1f}mm, Z={self.state.current_pos[2]*1000:.1f}mm")
            
        except Exception as e:
            print(f"❌ 移动到初始位置失败: {e}")
            print("⚠️  请手动确保机器人在安全位置")
    
    def convert_angles_to_robot(self, q_solver: List[float]) -> List[float]:
        """将求解器角度转换为实际机器人角度"""
        return [-angle for angle in q_solver]
    
    def convert_angles_from_robot(self, q_robot: List[float]) -> List[float]:
        """将实际机器人角度转换为求解器角度"""
        return [-angle for angle in q_robot]
    
    def _update_current_position(self):
        """更新当前末端位置"""
        q_solver = self.convert_angles_from_robot(self.state.current_q)
        T = self.robot_model.fkine(q_solver)
        self.state.current_pos = T.t.tolist()
    
    def solve_inverse_kinematics(self, target_pos: List[float]) -> Tuple[Optional[List[float]], bool]:
        """求解逆运动学"""
        try:
            # 创建目标变换矩阵
            T_target = SE3.Tx(target_pos[0]) * SE3.Ty(target_pos[1]) * SE3.Tz(target_pos[2])
            
            # 将当前角度转换为求解器角度作为初始位置
            current_q_solver = self.convert_angles_from_robot(self.state.current_q)
            
            # 求解逆运动学
            sol = self.robot_model.ikine_LM(
                T_target, 
                q0=current_q_solver, 
                mask=[1,1,1,0,0,0],  # 只约束位置，不约束姿态
                tol=1e-4, 
                joint_limits=True,
                slimit=20
            )
            
            if sol.success:
                # 转换为实际机器人角度
                q_solution = self.convert_angles_to_robot(sol.q.tolist())
                
                # 检查角度是否在允许范围内（±π/2）
                for i, angle in enumerate(q_solution):
                    if abs(angle) > np.pi/2:
                        print(f"❌ 求解的关节{i+1}角度超出范围: {np.degrees(angle):.1f}° (最大±90°)")
                        return None, False
                
                return q_solution, True
            else:
                return None, False
                
        except Exception as e:
            print(f"逆运动学求解失败: {e}")
            return None, False
    
    def move_to_position(self, target_pos: List[float], send_to_robot: bool = True) -> bool:
        """移动到目标位置"""
        if self.state.is_moving:
            return False
        
        self.state.is_moving = True
        self.state.target_pos = target_pos.copy()
        
        try:
            print(f"\n🎯 设定目标位置: X={target_pos[0]*1000:.1f}mm, Y={target_pos[1]*1000:.1f}mm, Z={target_pos[2]*1000:.1f}mm")
            
            # 求解逆运动学
            q_solution, success = self.solve_inverse_kinematics(target_pos)
            
            if success and q_solution is not None:
                self.state.current_q = q_solution
                self._update_current_position()
                
                # 显示实际到达位置
                actual_pos = self.state.current_pos
                print(f"📍 实际到达位置: X={actual_pos[0]*1000:.1f}mm, Y={actual_pos[1]*1000:.1f}mm, Z={actual_pos[2]*1000:.1f}mm")
                
                # 计算位置误差
                error = [abs(target_pos[i] - actual_pos[i]) for i in range(3)]
                total_error = sum(error)
                print(f"📏 位置误差: X={error[0]*1000:.2f}mm, Y={error[1]*1000:.2f}mm, Z={error[2]*1000:.2f}mm (总误差: {total_error*1000:.2f}mm)")
                
                # 显示关节角度
                q_degrees = [np.degrees(angle) for angle in q_solution]
                print(f"🔧 关节角度: [{', '.join([f'{angle:.1f}°' for angle in q_degrees])}]")
                
                # 发送到实际机器人
                if send_to_robot and self.real_robot is not None:
                    self._send_to_real_robot(q_solution)
                
                return True
            else:
                print(f"❌ 无法到达目标位置: X={target_pos[0]*1000:.1f}mm, Y={target_pos[1]*1000:.1f}mm, Z={target_pos[2]*1000:.1f}mm")
                return False
                
        except Exception as e:
            print(f"❌ 移动失败: {e}")
            return False
        finally:
            self.state.is_moving = False
    
    def _send_to_real_robot(self, q_angles: List[float]):
        """发送角度到实际机器人"""
        try:
            # 检查角度范围（±π/2）
            for i, angle in enumerate(q_angles):
                if abs(angle) > np.pi/2:
                    print(f"❌ 关节{i+1}角度超出范围: {np.degrees(angle):.1f}° (最大±90°)")
                    return
            
            # 转换角度单位（弧度到度数）
            q_degrees = [np.degrees(angle) for angle in q_angles]
            
            # 映射到机器人关节名称
            action = {
                "shoulder_pan.pos": q_degrees[0],
                "shoulder_lift.pos": q_degrees[1], 
                "elbow_flex.pos": q_degrees[2],
                "wrist_flex.pos": q_degrees[3],
            }
            
            if self.real_robot is not None:
                self.real_robot.send_action(action)
                print(f"✅ 已发送到机器人: {[f'{angle:.1f}°' for angle in q_degrees]}")
            else:
                print("❌ 机器人未连接")
            
        except Exception as e:
            print(f"❌ 发送到机器人失败: {e}")
    
    def keyboard_control_advanced(self):
        """高级键盘控制（使用keyboard库，需要root权限）"""
        if not KEYBOARD_AVAILABLE:
            return self.keyboard_control_pynput()
        
        print("\n=== 高级键盘控制模式（keyboard库）===")
        print("控制说明:")
        print("  W/S: 前进/后退 (X轴)")
        print("  A/D: 左/右移动 (Y轴)")
        print("  Q/E: 上升/下降 (Z轴)")
        print("  R: 重置到初始位置")
        print("  T: 切换自动发送模式")
        print("  SPACE: 手动发送当前位置")
        print("  ESC: 退出")
        print(f"步长: {self.step_size*1000:.1f}mm")
        print(f"自动发送: {'开' if self.auto_send else '关'}")
        
        self._print_status()
        
        # 注册按键事件
        keyboard.on_press_key('w', lambda _: self._move_axis(0, self.step_size))
        keyboard.on_press_key('s', lambda _: self._move_axis(0, -self.step_size))
        keyboard.on_press_key('a', lambda _: self._move_axis(1, -self.step_size))
        keyboard.on_press_key('d', lambda _: self._move_axis(1, self.step_size))
        keyboard.on_press_key('q', lambda _: self._move_axis(2, self.step_size))
        keyboard.on_press_key('e', lambda _: self._move_axis(2, -self.step_size))
        keyboard.on_press_key('r', lambda _: self._reset_position())
        keyboard.on_press_key('t', lambda _: self._toggle_auto_send())
        keyboard.on_press_key('space', lambda _: self._manual_send())
        keyboard.on_press_key('esc', lambda _: self._exit())
        
        # 保持程序运行
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            keyboard.unhook_all()
    
    def keyboard_control_pynput(self):
        """pynput键盘控制（无需root权限）"""
        if not PYNPUT_AVAILABLE:
            return self.keyboard_control_basic()
        
        print("\n=== 实时键盘控制模式（pynput库，无需root）===")
        print("控制说明:")
        print("  W/S: 前进/后退 (X轴)")
        print("  A/D: 左/右移动 (Y轴)")
        print("  Q/E: 上升/下降 (Z轴)")
        print("  R: 重置到初始位置")
        print("  T: 切换自动发送模式")
        print("  SPACE: 手动发送当前位置")
        print("  ESC: 退出")
        print(f"步长: {self.step_size*1000:.1f}mm")
        print(f"自动发送: {'开' if self.auto_send else '关'}")
        print("\n请确保终端窗口处于焦点状态")
        
        self._print_status()
        
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char:
                    char = key.char.lower()
                    if char == 'w':
                        self._move_axis(0, self.step_size)
                    elif char == 's':
                        self._move_axis(0, -self.step_size)
                    elif char == 'a':
                        self._move_axis(1, -self.step_size)
                    elif char == 'd':
                        self._move_axis(1, self.step_size)
                    elif char == 'q':
                        self._move_axis(2, self.step_size)
                    elif char == 'e':
                        self._move_axis(2, -self.step_size)
                    elif char == 'r':
                        self._reset_position()
                    elif char == 't':
                        self._toggle_auto_send()
                else:
                    # 处理特殊键
                    if key == pynput_keyboard.Key.space:
                        self._manual_send()
                    elif key == pynput_keyboard.Key.esc:
                        self._exit()
                        # 通过设置标志来停止监听，而不是返回False
                        self.running = False
            except AttributeError:
                pass
        
        # 启动键盘监听
        with pynput_keyboard.Listener(on_press=on_press) as listener:
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                listener.stop()
    
    def keyboard_control_basic(self):
        """基本键盘控制（使用input）"""
        print("\n=== 基本键盘控制模式 ===")
        print("控制说明:")
        print("  w: 前进 (X+)    s: 后退 (X-)")
        print("  a: 左移 (Y-)    d: 右移 (Y+)")
        print("  q: 上升 (Z+)    e: 下降 (Z-)")
        print("  r: 重置位置")
        print("  t: 切换自动发送")
        print("  p: 显示当前状态")
        print("  x: 退出")
        print(f"步长: {self.step_size*1000:.1f}mm")
        
        while self.running:
            self._print_status()
            
            try:
                cmd = input("请输入命令: ").strip().lower()
                
                if cmd == 'w':
                    self._move_axis(0, self.step_size)
                elif cmd == 's':
                    self._move_axis(0, -self.step_size)
                elif cmd == 'a':
                    self._move_axis(1, -self.step_size)
                elif cmd == 'd':
                    self._move_axis(1, self.step_size)
                elif cmd == 'q':
                    self._move_axis(2, self.step_size)
                elif cmd == 'e':
                    self._move_axis(2, -self.step_size)
                elif cmd == 'r':
                    self._reset_position()
                elif cmd == 't':
                    self._toggle_auto_send()
                elif cmd == 'p':
                    continue  # 只显示状态
                elif cmd == 'x':
                    break
                else:
                    print("❌ 无效命令")
                    
            except KeyboardInterrupt:
                break
    
    def _move_axis(self, axis: int, delta: float):
        """沿指定轴移动"""
        new_pos = self.state.target_pos.copy()
        new_pos[axis] += delta
        
        axis_names = ['X', 'Y', 'Z']
        print(f"\n🚀 尝试{axis_names[axis]}轴移动 {delta*1000:+.1f}mm")
        print(f"   从: X={self.state.target_pos[0]*1000:.1f}mm, Y={self.state.target_pos[1]*1000:.1f}mm, Z={self.state.target_pos[2]*1000:.1f}mm")
        print(f"   到: X={new_pos[0]*1000:.1f}mm, Y={new_pos[1]*1000:.1f}mm, Z={new_pos[2]*1000:.1f}mm")
        
        # 检查工作空间限制
        if self._check_workspace_limits(new_pos):
            if self.move_to_position(new_pos, self.auto_send):
                print(f"✅ {axis_names[axis]}轴移动成功")
            else:
                print(f"❌ 移动失败")
        else:
            print(f"❌ 超出工作空间限制")
    
    def _check_workspace_limits(self, pos: List[float]) -> bool:
        """检查位置是否在工作空间内（考虑±π/2角度限制）"""
        x, y, z = pos
        
        # 基于±90度关节角度限制的实际可达范围
        # 调整后包含0度初始位置 (315.10, 0.00, 170.20)mm
        if not (0.0 <= x <= 0.35):  # X轴范围（包含初始位置）
            print(f"❌ X轴超出范围: {x:.3f}m (允许范围: 0.08-0.35m)")
            return False
        if not (-0.4 <= y <= 0.4):   # Y轴范围（对称）
            print(f"❌ Y轴超出范围: {y:.3f}m (允许范围: -0.20-0.20m)")
            return False
        if not (0.0 <= z <= 0.4):   # Z轴范围（包含初始位置）
            print(f"❌ Z轴超出范围: {z:.3f}m (允许范围: 0.08-0.30m)")
            return False
        
        # 检查是否超出最大reach（基于实际测试）
        reach = np.sqrt(x**2 + y**2 + z**2)
        if reach > 0.43:  # 扩大reach限制到430mm
            print(f"❌ 超出最大reach: {reach:.3f}m (最大: 0.45m)")
            return False
        
        return True
    
    def _reset_position(self):
        """重置到初始位置（所有关节0度）"""
        print(f"\n🔄 重置到初始位置")
        print(f"   当前位置: X={self.state.current_pos[0]*1000:.1f}mm, Y={self.state.current_pos[1]*1000:.1f}mm, Z={self.state.current_pos[2]*1000:.1f}mm")
        
        # 直接将所有关节设置为0度
        initial_q = [0.0, 0.0, 0.0, 0.0]
        
        try:
            # 更新状态
            self.state.current_q = initial_q
            self._update_current_position()
            self.state.target_pos = self.state.current_pos.copy()
            
            print(f"🎯 目标位置: 初始位置（所有关节0度）")
            print(f"📍 实际位置: X={self.state.current_pos[0]*1000:.1f}mm, Y={self.state.current_pos[1]*1000:.1f}mm, Z={self.state.current_pos[2]*1000:.1f}mm")
            print(f"🔧 关节角度: [0.0°, 0.0°, 0.0°, 0.0°]")
            
            # 发送到真实机器人
            if self.real_robot is not None and self.auto_send:
                self._send_to_real_robot(initial_q)
                print(f"✅ 已发送到机器人")
            else:
                print(f"✅ 重置完成（仅仿真）")
            
        except Exception as e:
            print(f"❌ 重置失败: {e}")
    
    def _toggle_auto_send(self):
        """切换自动发送模式"""
        self.auto_send = not self.auto_send
        status = "开启" if self.auto_send else "关闭"
        print(f"🔄 自动发送模式: {status}")
    
    def _manual_send(self):
        """手动发送当前位置"""
        if self.real_robot is not None:
            self._send_to_real_robot(self.state.current_q)
        else:
            print("❌ 实际机器人未连接")
    
    def _exit(self):
        """退出程序"""
        self.running = False
        print("👋 退出程序")
    
    def _print_status(self):
        """打印当前状态"""
        print(f"\n=== 当前状态 ===")
        print(f"🎯 设定目标位置: X={self.state.target_pos[0]*1000:.1f}mm, Y={self.state.target_pos[1]*1000:.1f}mm, Z={self.state.target_pos[2]*1000:.1f}mm")
        print(f"📍 机器人实际位置: X={self.state.current_pos[0]*1000:.1f}mm, Y={self.state.current_pos[1]*1000:.1f}mm, Z={self.state.current_pos[2]*1000:.1f}mm")
        
        # 计算位置误差
        error = [abs(self.state.target_pos[i] - self.state.current_pos[i]) for i in range(3)]
        total_error = sum(error)
        print(f"📏 位置误差: X={error[0]*1000:.2f}mm, Y={error[1]*1000:.2f}mm, Z={error[2]*1000:.2f}mm (总误差: {total_error*1000:.2f}mm)")
        
        print(f"🔧 当前关节角度: [{', '.join([f'{np.degrees(angle):.1f}°' for angle in self.state.current_q])}]")
        print(f"🔄 自动发送模式: {'开启' if self.auto_send else '关闭'}")
        print(f"🤖 机器人连接状态: {'已连接' if self.real_robot is not None else '未连接（仅仿真）'}")
        print(f"📐 移动步长: {self.step_size*1000:.1f}mm")
    
    def run(self):
        """运行控制器"""
        print("=== SO101机器人键盘控制 ===")
        
        # 显示安全提示
        if self.real_robot is not None:
            print("\n⚠️  安全提示:")
            print("   - 机器人已自动移动到安全的初始位置（所有关节0度）")
            print("   - 所有关节运动范围限制在±90度")
            print("   - 请确保机器人周围无障碍物")
            input("   按回车键继续...")
        
        # 优先级: pynput库（无需root） > keyboard库（需要root） > 基本输入模式
        if PYNPUT_AVAILABLE:
            self.keyboard_control_pynput()
        elif KEYBOARD_AVAILABLE:
            try:
                self.keyboard_control_advanced()
            except (PermissionError, OSError) as e:
                print(f"\n❌ keyboard库错误: {e}")
                print("请使用 sudo 运行或切换到基本输入模式")
                self.keyboard_control_basic()
        else:
            self.keyboard_control_basic()
    
    def cleanup(self):
        """清理资源"""
        if self.real_robot is not None:
            try:
                self.real_robot.disconnect()
                print("✅ 机器人连接已断开")
            except:
                pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SO101机器人键盘控制")
    parser.add_argument("--port", default="/dev/ttyACM0", help="机器人串口")
    parser.add_argument("--step", type=float, default=0.002, help="移动步长(米)")
    parser.add_argument("--no-robot", action="store_true", help="仅仿真模式，不连接实际机器人")
    parser.add_argument("--basic-mode", action="store_true", help="强制使用基本输入模式（无需root权限）")
    
    args = parser.parse_args()
    
    controller = None
    try:
        # 如果指定仅仿真模式，禁用实际机器人
        if args.no_robot:
            global ROBOT_AVAILABLE
            ROBOT_AVAILABLE = False
        
        # 如果指定基本模式，禁用keyboard库
        if args.basic_mode:
            global KEYBOARD_AVAILABLE
            KEYBOARD_AVAILABLE = False
        
        controller = KeyboardController(
            robot_port=args.port, 
            step_size=args.step
        )
        controller.run()
        
    except KeyboardInterrupt:
        print("\n👋 程序被中断")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
    finally:
        if controller:
            controller.cleanup()


if __name__ == "__main__":
    main() 