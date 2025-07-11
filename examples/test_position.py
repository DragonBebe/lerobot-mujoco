#!/usr/bin/env python3
"""
位置测试脚本 - test_position.py
测试特定位置的IK求解，验证修复效果
"""

import subprocess
import sys

def test_position(x, y, z, description=""):
    """测试指定位置"""
    print(f"🧪 测试位置: X={x*1000:.0f}mm, Y={y*1000:.0f}mm, Z={z*1000:.0f}mm {description}")
    
    # 准备输入
    input_str = f"{x} {y} {z}\nq\n"
    
    try:
        # 运行程序
        result = subprocess.run(
            ["conda", "run", "-n", "lerobot", "python", "8-DH-IK.py"],
            input=input_str,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        # 分析输出
        output = result.stdout
        
        if "工作空间检查通过" in output:
            print("✅ 工作空间检查通过")
        else:
            print("❌ 工作空间检查失败")
        
        if "目标关节角度:" in output:
            print("✅ IK求解成功")
        else:
            print("❌ IK求解失败")
            
        if "关节角度安全设置完成" in output:
            print("✅ 关节角度设置成功")
        else:
            print("❌ 关节角度设置失败")
            
        if "Segmentation fault" in output or "core dumped" in output or result.returncode != 0:
            print("❌ 程序崩溃")
            if result.stderr:
                print(f"错误: {result.stderr}")
        else:
            print("✅ 程序运行正常")
            
        print("-" * 50)
        return "✅" in output
        
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        print("-" * 50)
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("-" * 50)
        return False

def main():
    """主测试函数"""
    print("🔧 位置测试开始")
    print("=" * 60)
    
    # 测试位置列表
    test_cases = [
        ([0.0, 0.3, 0.2], "- 之前出错的位置"),
        ([0.0, -0.3, 0.2], "- 另一侧对称位置"), 
        ([0.2, 0.0, 0.1], "- 前方安全位置"),
        ([0.25, 0.0, 0.05], "- 红色cube位置"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for position, description in test_cases:
        if test_position(position[0], position[1], position[2], description):
            passed += 1
    
    print(f"📊 测试总结: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！IK系统工作正常")
    elif passed >= total * 0.75:
        print("✅ 大部分测试通过，系统基本正常")
    else:
        print("⚠️ 多个测试失败，需要进一步检查")

if __name__ == "__main__":
    main() 