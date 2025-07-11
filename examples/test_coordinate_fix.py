#!/usr/bin/env python3
"""
坐标系修复验证脚本 - test_coordinate_fix.py

测试Y轴坐标系转换修复是否正确
"""

import subprocess
import sys
import time

def test_coordinate_fix():
    """测试坐标系修复"""
    print("🧪 测试坐标系修复效果")
    print("=" * 50)
    
    # 测试用例：Y轴正负方向
    test_cases = [
        ([0, 0.3, 0.2], "Y轴正方向 (期望：机械臂向前移动)"),
        ([0, -0.3, 0.2], "Y轴负方向 (期望：机械臂向后移动)"),
        ([0.2, 0, 0.15], "X轴正方向"),
        ([0, 0, 0.3], "Z轴正方向 (垂直向上)"),
    ]
    
    for i, (pos, description) in enumerate(test_cases):
        print(f"\n📋 测试 {i+1}: {description}")
        print(f"   输入位置: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}")
        
        # 准备输入字符串
        input_str = f"{pos[0]} {pos[1]} {pos[2]}\npos\nq\n"
        
        try:
            # 运行修复后的程序
            result = subprocess.run(
                ["conda", "run", "-n", "lerobot", "python", "8-DH-IK.py"],
                input=input_str,
                text=True,
                capture_output=True,
                timeout=20
            )
            
            output = result.stdout
            
            # 分析输出
            if "坐标系转换:" in output:
                print("   ✅ 坐标系转换功能正常")
                # 提取转换信息
                for line in output.split('\n'):
                    if "坐标系转换:" in line:
                        print(f"   🔧 {line.strip()}")
            else:
                print("   ❌ 未找到坐标系转换信息")
            
            if "IK验证通过" in output:
                print("   ✅ IK验证通过")
            elif "IK验证失败" in output:
                print("   ❌ IK验证失败")
            
            if "FK误差:" in output:
                # 提取FK误差信息
                for line in output.split('\n'):
                    if "FK误差:" in line:
                        print(f"   📏 {line.strip()}")
            
            # 检查是否有内存错误
            if "mj_stackAlloc" in output or "Segmentation fault" in result.stderr:
                print("   ❌ 仍有内存错误")
            else:
                print("   ✅ 无内存错误")
                
        except subprocess.TimeoutExpired:
            print("   ⏰ 测试超时")
        except Exception as e:
            print(f"   ❌ 测试错误: {e}")
        
        print("-" * 30)
        time.sleep(2)  # 间隔时间避免过快

if __name__ == "__main__":
    test_coordinate_fix() 