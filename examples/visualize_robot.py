#!/usr/bin/env python3
"""
SO101四轴机器人可视化
显示机器人的初始姿态
"""

import numpy as np
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
    print("成功导入可视化库!")
except ImportError as e:
    print(f"请安装可视化库: pip install roboticstoolbox-python")
    print(f"错误信息: {e}")
    exit(1)


def convert_angles_to_robot(q_solver):
    """
    将求解器输出的角度转换为实际机器人角度
    由于Z轴方向与右手定则相反，所有关节角度都需要取负号
    """
    q_robot = [-angle for angle in q_solver]
    return q_robot


def convert_angles_from_robot(q_robot):
    """
    将实际机器人角度转换为求解器角度
    """
    q_solver = [-angle for angle in q_robot]
    return q_solver


def create_so101_robot():
    """创建SO101四轴机器人 - 使用经典DH参数"""
    robot = rtb.DHRobot([
        rtb.RevoluteDH(a=0.0304, alpha=np.pi/2, d=0.0542, offset=0),  # 关节1
        rtb.RevoluteDH(a=0.116, alpha=0, d=0, offset=np.pi/2),        # 关节2
        rtb.RevoluteDH(a=0.1347, alpha=0, d=0, offset=-np.pi/2),      # 关节3
        rtb.RevoluteDH(a=0.15, alpha=0, d=0, offset=0)               # 关节4
    ], name="SO101 Four-Axis Robot")
    
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


def show_robot_info(robot, q=None):
    """显示机器人信息"""
    if q is None:
        q = [0, 0, 0, 0]  # 零位姿态
    
    print(f"\n=== 机器人信息 ===")
    print(f"实际机器人角度 (弧度): {q}")
    print(f"实际机器人角度 (度数): {[f'{np.degrees(angle):.2f}°' for angle in q]}")
    
    # 转换为求解器角度进行运动学计算
    q_solver = convert_angles_from_robot(q)
    print(f"求解器角度 (弧度): {q_solver}")
    
    # 计算末端位置
    T = robot.fkine(q_solver)
    print(f"末端位置: x={T.t[0]:.4f}, y={T.t[1]:.4f}, z={T.t[2]:.4f}")
    
    # 计算各关节位置
    print(f"\n各关节位置:")
    for i in range(len(q)):
        # 创建子机器人来计算前i+1个关节的位置
        sub_robot = rtb.DHRobot(robot.links[:i+1])
        T_joint = sub_robot.fkine(q_solver[:i+1])
        print(f"  关节{i+1}: x={T_joint.t[0]:.4f}, y={T_joint.t[1]:.4f}, z={T_joint.t[2]:.4f}")


def solve_inverse_kinematics(robot, target_pos, current_q=None):
    """求解逆运动学"""
    if current_q is None:
        current_q = [0, 0, 0, 0]  # 默认初始位置
    
    print(f"\n=== 逆运动学求解 ===")
    print(f"目标位置: x={target_pos[0]:.4f}, y={target_pos[1]:.4f}, z={target_pos[2]:.4f}")
    
    # 创建目标变换矩阵（只约束位置，不约束姿态）
    from spatialmath import SE3
    T_target = SE3.Tx(target_pos[0]) * SE3.Ty(target_pos[1]) * SE3.Tz(target_pos[2])
    
    try:
        # 使用多种方法尝试求解
        methods = ['NR', 'LM']  # Newton-Raphson, Levenberg-Marquardt
        
        for method in methods:
            try:
                # 将当前机器人角度转换为求解器角度
                current_q_solver = convert_angles_from_robot(current_q)
                # 求解逆运动学（只约束位置）
                sol = robot.ikine_LM(T_target, q0=current_q_solver, mask=[1,1,1,0,0,0], tol=1e-4, joint_limits=True)
                
                if sol.success:
                    q_solution_solver = sol.q
                    
                    # 转换为实际机器人角度
                    q_solution = convert_angles_to_robot(q_solution_solver)
                    
                    # 验证解的有效性（使用求解器角度验证）
                    T_verify = robot.fkine(q_solution_solver)
                    pos_error = np.linalg.norm(T_verify.t - target_pos)
                    
                    print(f"✅ 求解成功 (方法: {method})")
                    print(f"求解器角度 (弧度): {[f'{angle:.4f}' for angle in q_solution_solver]}")
                    print(f"实际机器人角度 (弧度): {[f'{angle:.4f}' for angle in q_solution]}")
                    print(f"实际机器人角度 (度数): {[f'{np.degrees(angle):.2f}°' for angle in q_solution]}")
                    print(f"实际到达位置: x={T_verify.t[0]:.4f}, y={T_verify.t[1]:.4f}, z={T_verify.t[2]:.4f}")
                    print(f"位置误差: {pos_error*1000:.2f} mm")
                    
                    return q_solution, True
                    
            except Exception as e:
                print(f"方法 {method} 失败: {e}")
                continue
        
        print("❌ 所有求解方法都失败了")
        return current_q, False
        
    except Exception as e:
        print(f"❌ 逆运动学求解失败: {e}")
        return current_q, False


def visualize_robot_simple(robot, q=None):
    """简单可视化机器人"""
    if q is None:
        q = [0, 0, 0, 0]  # 零位姿态
    
    print(f"\n=== 机器人可视化 ===")
    print(f"正在显示机器人姿态...")
    print(f"实际机器人角度: {[f'{np.degrees(angle):.2f}°' for angle in q]}")
    
    # 转换为求解器角度用于可视化
    q_solver = convert_angles_from_robot(q)
    print(f"求解器角度: {[f'{np.degrees(angle):.2f}°' for angle in q_solver]}")
    
    try:
        # 使用Robotics Toolbox的简单可视化，保持窗口打开
        robot.plot(q_solver, block=True)
        print("✅ 可视化窗口已打开")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        print("请确保有图形界面支持")


def interactive_ik_demo(robot):
    """交互式逆运动学演示"""
    print(f"\n=== 交互式逆运动学演示 ===")
    print("输入目标位置，机器人将尝试到达该位置")
    print("输入 'q' 退出")
    
    current_q = [0, 0, 0, 0]  # 当前关节角度
    
    while True:
        try:
            print(f"\n当前实际机器人角度: {[f'{np.degrees(angle):.2f}°' for angle in current_q]}")
            current_q_solver = convert_angles_from_robot(current_q)
            T_current = robot.fkine(current_q_solver)
            print(f"当前末端位置: x={T_current.t[0]:.4f}, y={T_current.t[1]:.4f}, z={T_current.t[2]:.4f}")
            
            # 获取用户输入
            user_input = input("\n请输入目标位置 (格式: x y z) 或 'q' 退出: ").strip()
            
            if user_input.lower() == 'q':
                break
                
            # 解析坐标
            coords = user_input.split()
            if len(coords) != 3:
                print("❌ 请输入3个坐标值 (x y z)")
                continue
                
            try:
                target_pos = [float(coords[0]), float(coords[1]), float(coords[2])]
            except ValueError:
                print("❌ 请输入有效的数字")
                continue
            
            # 求解逆运动学
            q_solution, success = solve_inverse_kinematics(robot, target_pos, current_q)
            
            if success:
                current_q = q_solution
                
                # 询问是否可视化
                viz_input = input("是否可视化这个姿态? (y/n): ").strip().lower()
                if viz_input == 'y' or viz_input == 'yes':
                    visualize_robot_simple(robot, current_q)
            else:
                print("无法到达目标位置，保持当前姿态")
                
        except KeyboardInterrupt:
            print("\n👋 程序已退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def preset_positions_demo(robot):
    """预设位置演示"""
    print(f"\n=== 预设位置演示 ===")
    
    # 定义一些预设的目标位置
    preset_positions = [
        ([0.15, 0, 0.15], "正前方"),
        ([0.12, 0.08, 0.18], "右前方"),
        ([0.12, -0.08, 0.18], "左前方"),
        ([0.08, 0, 0.25], "正上方"),
        ([0.20, 0, 0.10], "正前低位"),
    ]
    
    current_q = [0, 0, 0, 0]
    
    for i, (target_pos, description) in enumerate(preset_positions):
        print(f"\n--- 预设位置 {i+1}: {description} ---")
        
        # 求解逆运动学
        q_solution, success = solve_inverse_kinematics(robot, target_pos, current_q)
        
        if success:
            current_q = q_solution
            
            # 询问是否可视化
            viz_input = input("是否可视化这个姿态? (y/n): ").strip().lower()
            if viz_input == 'y' or viz_input == 'yes':
                visualize_robot_simple(robot, current_q)
        else:
            print(f"无法到达位置: {description}")
        
        # 询问是否继续
        continue_input = input("继续下一个位置? (y/n): ").strip().lower()
        if continue_input != 'y' and continue_input != 'yes':
            break


def calculate_workspace(robot, n_samples=500):
    """计算机器人工作空间"""
    print(f"\n=== 工作空间分析 ===")
    workspace_points = []
    
    # 随机采样关节角度
    for _ in range(n_samples):
        q = []
        for link in robot.links:
            qmin, qmax = link.qlim
            q.append(np.random.uniform(qmin, qmax))
        
        try:
            # 转换为求解器角度进行运动学计算
            q_solver = convert_angles_from_robot(q)
            T = robot.fkine(q_solver)
            workspace_points.append(T.t)
        except:
            continue
    
    if workspace_points:
        workspace_points = np.array(workspace_points)
        print(f"工作空间范围:")
        print(f"  X: {workspace_points[:, 0].min():.3f} ~ {workspace_points[:, 0].max():.3f} m")
        print(f"  Y: {workspace_points[:, 1].min():.3f} ~ {workspace_points[:, 1].max():.3f} m")
        print(f"  Z: {workspace_points[:, 2].min():.3f} ~ {workspace_points[:, 2].max():.3f} m")
        print(f"  最大reach: {np.max(np.linalg.norm(workspace_points, axis=1)):.3f} m")
    
    return workspace_points


def main():
    """主函数"""
    print("=== SO101四轴机器人可视化 ===")
    
    # 创建机器人
    robot = create_so101_robot()
    print(f"机器人创建成功: {robot.name}")
    
    # 显示机器人参数
    print(f"\n机器人参数:")
    print(robot)
    
    # 显示零位信息
    show_robot_info(robot, [0, 0, 0, 0])
    
    # 显示其他姿态信息
    poses = [
        ([0, 0, 0, 0], "零位姿态"),
        ([0, np.pi/4, -np.pi/4, 0], "前伸姿态"),
        ([np.pi/2, 0, np.pi/2, 0], "侧伸姿态"),
        ([0, -np.pi/6, np.pi/3, np.pi/6], "上举姿态"),
    ]
    
    for q, title in poses:
        print(f"\n--- {title} ---")
        show_robot_info(robot, q)
    
    # 计算工作空间
    calculate_workspace(robot)
    
    # 功能选择菜单
    while True:
        print(f"\n=== 功能菜单 ===")
        print("1. 显示零位可视化")
        print("2. 交互式逆运动学演示")
        print("3. 预设位置演示")
        print("4. 退出")
        
        try:
            choice = input("请选择功能 (1-4): ").strip()
            
            if choice == '1':
                visualize_robot_simple(robot)
            elif choice == '2':
                interactive_ik_demo(robot)
            elif choice == '3':
                preset_positions_demo(robot)
            elif choice == '4':
                print("👋 程序已退出")
                break
            else:
                print("❌ 请输入有效的选项 (1-4)")
                
        except KeyboardInterrupt:
            print("\n👋 程序已退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main() 